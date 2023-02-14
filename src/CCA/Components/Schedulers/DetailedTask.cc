/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/Schedulers/DependencyBatch.h>
#include <CCA/Components/Schedulers/DetailedTask.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/SchedulerCommon.h>

#if defined(UINTAH_ENABLE_DEVICE)
#include <CCA/Components/Schedulers/GPUMemoryPool.hpp>
#endif

#include <Core/Containers/ConsecutiveRangeSet.h>
#include <Core/Parallel/MasterLock.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Util/DOUT.hpp>

#include <sci_defs/config_defs.h>

#include <sstream>
#include <string>

using namespace Uintah;

// declared in DetailedTasks.cc - used in both places to protect external ready
// queue (hence, extern here)
namespace Uintah {
extern Uintah::MasterLock g_external_ready_mutex;
extern Dout g_scrubbing_dbg;
extern std::string g_var_scrub_dbg;
extern int g_patch_scrub_dbg;
} // namespace Uintah

namespace {
Uintah::MasterLock g_internal_dependency_mutex{};
Uintah::MasterLock g_dtask_output_mutex{};

Dout g_internal_deps_dbg("InternalDeps", "DetailedTask",
                         "info on internal (intra-nodal) data dependencies",
                         false);
Dout g_external_deps_dbg("ExternalDeps", "DetailedTask",
                         "info on external (inter-nodal) data dependencies",
                         false);
} // namespace

//_____________________________________________________________________________
//
DetailedTask::DetailedTask(Task *task, const PatchSubset *patches,
                           const MaterialSubset *matls,
                           DetailedTasks *taskGroup)
    : m_task(task), m_patches(patches), m_matls(matls),
      m_task_group(taskGroup) {
  if (m_patches) {
    // patches and matls must be sorted
    ASSERT(std::is_sorted(m_patches->getVector().begin(),
                          m_patches->getVector().end(), Patch::Compare()));
    m_patches->addReference();
  }
  if (m_matls) {
    // patches and matls must be sorted
    ASSERT(std::is_sorted(m_matls->getVector().begin(),
                          m_matls->getVector().end()));
    m_matls->addReference();
  }
}

//_____________________________________________________________________________
//
DetailedTask::~DetailedTask() {
  if (m_patches && m_patches->removeReference()) {
    delete m_patches;
  }

  if (m_matls && m_matls->removeReference()) {
    delete m_matls;
  }
}

//_____________________________________________________________________________
//
void DetailedTask::doit(const ProcessorGroup *pg,
                        std::vector<OnDemandDataWarehouseP> &oddws,
                        std::vector<DataWarehouseP> &dws,
                        Task::CallBackEvent event /* = Task::CallBackEvent::CPU */
) {
  // stop timing the task wait
  m_wait_timer.stop();

  // start timing the execution duration
  m_exec_timer.start();
  //__________________________________
  //  Debugging output
  if (g_internal_deps_dbg) {
    std::ostringstream message;
    message << "DetailedTask " << this << " begin doit()\n";
    message << " task is " << m_task << "\n";
    message << "   num Pending Deps: " << m_num_pending_internal_dependencies
            << "\n";
    message << "   Originally needed deps (" << m_internal_dependencies.size()
            << "):\n";

    auto iter = m_internal_dependencies.begin();
    for (std::size_t i = 0u; iter != m_internal_dependencies.end();
         ++iter, ++i) {
      message << i << ":    " << *((*iter).m_prerequisite_task->getTask())
              << "\n";
    }

    DOUT(true, message.str());
  }

  for (std::size_t i = 0; i < dws.size(); ++i) {
    if (oddws[i] != nullptr) {
      oddws[i]->pushRunningTask(m_task, &oddws);
    }
  }

  // Determine if task will be executed on CPU or GPU
  if (m_task->usesDevice()) {
#if defined(UINTAH_ENABLE_DEVICE)
    // Run the GPU task  Technically the engine has structure to run one task on
    // multiple devices if that task had patches on multiple devices  So run the
    // task once per device. As often as possible, we want to design tasks so
    // each task runs on only one device, instead of a one to many
    // relationship.

    for (auto currentDeviceID : deviceNums_) {
      gpuSetDevice(currentDeviceID);
      GPUDataWarehouse *host_oldtaskdw = getTaskGpuDataWarehouse(currentDeviceID, Task::WhichDW::OldDW);
      GPUDataWarehouse *device_oldtaskdw = nullptr;
      if (host_oldtaskdw) {
        device_oldtaskdw = host_oldtaskdw->getdevice_ptr();
      }
      GPUDataWarehouse *host_newtaskdw = getTaskGpuDataWarehouse(currentDeviceID, Task::WhichDW::NewDW);
      GPUDataWarehouse *device_newtaskdw = nullptr;
      if (host_newtaskdw) {
        device_newtaskdw = host_newtaskdw->getdevice_ptr();
      }
      m_task->doit(this, event, pg, m_patches, m_matls, dws, device_oldtaskdw,
                   device_newtaskdw);
    }
#endif
  } else {
    m_task->doit(this, event, pg, m_patches, m_matls, dws, nullptr, nullptr);
  }

  for (std::size_t i = 0u; i < dws.size(); ++i) {
    if (oddws[i] != nullptr) {
      oddws[i]->checkTasksAccesses(m_patches, m_matls);
      oddws[i]->popRunningTask();
    }
  }
}

//_____________________________________________________________________________
//
void DetailedTask::scrub(std::vector<OnDemandDataWarehouseP> &dws) {
  DOUTR(g_scrubbing_dbg,
        "  DetailedTask::Scrub, Starting scrub after task: " << *this);

  const Task *task = getTask();

  const std::set<const VarLabel *, VarLabel::Compare> &initialRequires =
      m_task_group->getSchedulerCommon()->getInitialRequiredVars();
  const std::set<std::string> &unscrubbables =
      m_task_group->getSchedulerCommon()->getNoScrubVars();

  // Decrement the scrub count for each of the required variables
  for (const Task::Dependency *req = task->getRequires(); req != nullptr;
       req = req->m_next) {
    TypeDescription::Type type = req->m_var->typeDescription()->getType();
    Patch::VariableBasis basis = Patch::translateTypeToBasis(type, false);
    if (type != TypeDescription::ReductionVariable &&
        type != TypeDescription::SoleVariable) {
      int dw = req->mapDataWarehouse();

      DataWarehouse::ScrubMode scrubmode = dws[dw]->getScrubMode();
      if (scrubmode == DataWarehouse::ScrubComplete ||
          (scrubmode == DataWarehouse::ScrubNonPermanent &&
           initialRequires.find(req->m_var) == initialRequires.end())) {

        if (unscrubbables.find(req->m_var->getName()) != unscrubbables.end())
          continue;

        constHandle<PatchSubset> patches =
            req->getPatchesUnderDomain(getPatches());
        constHandle<MaterialSubset> matls =
            req->getMaterialsUnderDomain(getMaterials());
        for (int i = 0; i < patches->size(); i++) {
          const Patch *patch = patches->get(i);
          Patch::selectType neighbors;
          IntVector low, high;

          if (req->m_patches_dom == Task::PatchDomainSpec::CoarseLevel ||
              req->m_patches_dom == Task::PatchDomainSpec::FineLevel ||
              req->m_num_ghost_cells == 0) {
            // we already have the right patches
            neighbors.push_back(patch);
          } else {
            patch->computeVariableExtents(type, req->m_var->getBoundaryLayer(),
                                          req->m_gtype, req->m_num_ghost_cells,
                                          neighbors, low, high);
          }

          for (unsigned int i = 0; i < neighbors.size(); i++) {
            const Patch *neighbor = neighbors[i];

            if (req->m_patches_dom == Task::PatchDomainSpec::ThisLevel && patch != neighbor) {
              // don't scrub on AMR overlapping patches...
              IntVector l = Max(neighbor->getExtraLowIndex(
                                    basis, req->m_var->getBoundaryLayer()),
                                low);
              IntVector h = Min(neighbor->getExtraHighIndex(
                                    basis, req->m_var->getBoundaryLayer()),
                                high);

              patch->cullIntersection(basis, req->m_var->getBoundaryLayer(),
                                      neighbor->getRealPatch(), l, h);

              if (l == h) {
                continue;
              }
            }

            if (req->m_patches_dom == Task::PatchDomainSpec::FineLevel) {
              // don't count if it only overlaps extra cells
              IntVector l = patch->getExtraLowIndex(basis, IntVector(0, 0, 0)),
                        h = patch->getExtraHighIndex(basis, IntVector(0, 0, 0));
              IntVector fl = neighbor->getLowIndex(basis),
                        fh = neighbor->getHighIndex(basis);
              IntVector il = Max(l, neighbor->getLevel()->mapCellToCoarser(fl));
              IntVector ih = Min(h, neighbor->getLevel()->mapCellToCoarser(fh));
              if (ih.x() <= il.x() || ih.y() <= il.y() || ih.z() <= il.z()) {
                continue;
              }
            }

            for (int m = 0; m < matls->size(); m++) {
              int count;
              try {
                // there are a few rare cases in an AMR framework where you
                // require from an OldDW, but only ones internal to the W-cycle
                // (and not the previous timestep) which can have variables not
                // exist in the OldDW.
                count = dws[dw]->decrementScrubCount(req->m_var, matls->get(m),
                                                     neighbor);
                if (g_scrubbing_dbg &&
                    (req->m_var->getName() == g_var_scrub_dbg ||
                     g_var_scrub_dbg == "") &&
                    (neighbor->getID() == g_patch_scrub_dbg ||
                     g_patch_scrub_dbg == -1)) {

                  DOUTR(g_scrubbing_dbg,
                        "    decrementing scrub count for requires of "
                            << dws[dw]->getID() << "/" << neighbor->getID()
                            << "/" << matls->get(m) << "/"
                            << req->m_var->getName() << ": " << count
                            << (count == 0 ? " - scrubbed\n" : "\n"));
                }
              } catch (UnknownVariable &e) {
                std::cerr << "   BAD BOY FROM Task : " << *this << " scrubbing "
                          << *req << " PATCHES: " << *patches.get_rep()
                          << std::endl;
                throw e;
              }
            }
          }
        }
      }
    }
  } // end for req

  // Scrub modifies
  for (const Task::Dependency *mod = task->getModifies(); mod != nullptr;
       mod = mod->m_next) {
    int dw = mod->mapDataWarehouse();
    DataWarehouse::ScrubMode scrubmode = dws[dw]->getScrubMode();

    if (scrubmode == DataWarehouse::ScrubComplete ||
        (scrubmode == DataWarehouse::ScrubNonPermanent &&
         initialRequires.find(mod->m_var) == initialRequires.end())) {

      if (unscrubbables.find(mod->m_var->getName()) != unscrubbables.end())
        continue;

      constHandle<PatchSubset> patches =
          mod->getPatchesUnderDomain(getPatches());
      constHandle<MaterialSubset> matls =
          mod->getMaterialsUnderDomain(getMaterials());
      TypeDescription::Type type = mod->m_var->typeDescription()->getType();

      if (type != TypeDescription::ReductionVariable &&
          type != TypeDescription::SoleVariable) {
        for (int i = 0; i < patches->size(); i++) {
          const Patch *patch = patches->get(i);

          for (int m = 0; m < matls->size(); m++) {
            int count =
                dws[dw]->decrementScrubCount(mod->m_var, matls->get(m), patch);

            if (g_scrubbing_dbg &&
                (mod->m_var->getName() == g_var_scrub_dbg ||
                 g_var_scrub_dbg == "") &&
                (patch->getID() == g_patch_scrub_dbg ||
                 g_patch_scrub_dbg == -1)) {
              DOUTR(g_scrubbing_dbg,
                    "    decrementing scrub count for modifies of "
                        << dws[dw]->getID() << "/" << patch->getID() << "/"
                        << matls->get(m) << "/" << mod->m_var->getName() << ": "
                        << count << (count == 0 ? " - scrubbed\n" : "\n"));
            }
          }
        }
      }
    }
  }

  // Set the scrub count for each of the computes variables
  for (const Task::Dependency *comp = task->getComputes(); comp != nullptr;
       comp = comp->m_next) {
    TypeDescription::Type type = comp->m_var->typeDescription()->getType();

    if (type != TypeDescription::ReductionVariable &&
        type != TypeDescription::SoleVariable) {
      Task::WhichDW whichdw = comp->m_whichdw;
      int dw = comp->mapDataWarehouse();
      DataWarehouse::ScrubMode scrubmode = dws[dw]->getScrubMode();

      if (scrubmode == DataWarehouse::ScrubComplete ||
          (scrubmode == DataWarehouse::ScrubNonPermanent &&
           initialRequires.find(comp->m_var) == initialRequires.end())) {
        constHandle<PatchSubset> patches =
            comp->getPatchesUnderDomain(getPatches());
        constHandle<MaterialSubset> matls =
            comp->getMaterialsUnderDomain(getMaterials());

        if (unscrubbables.find(comp->m_var->getName()) != unscrubbables.end()) {
          continue;
        }

        for (int i = 0; i < patches->size(); i++) {
          const Patch *patch = patches->get(i);

          for (int m = 0; m < matls->size(); m++) {
            int matl = matls->get(m);
            int count;

            if (m_task_group->getScrubCount(comp->m_var, matl, patch, static_cast<int>(whichdw),
                                            count)) {

              if (g_scrubbing_dbg &&
                  (comp->m_var->getName() == g_var_scrub_dbg ||
                   g_var_scrub_dbg == "") &&
                  (patch->getID() == g_patch_scrub_dbg ||
                   g_patch_scrub_dbg == -1)) {
                DOUTR(true, "    setting scrub count for computes of "
                                << dws[dw]->getID() << "/" << patch->getID()
                                << "/" << matls->get(m) << "/"
                                << comp->m_var->getName() << ": " << count);
              }
              dws[dw]->setScrubCount(comp->m_var, matl, patch, count);
            } else {
              // Not in the scrub map, must be never needed...
              if (g_scrubbing_dbg &&
                  (comp->m_var->getName() == g_var_scrub_dbg ||
                   g_var_scrub_dbg == "") &&
                  (patch->getID() == g_patch_scrub_dbg ||
                   g_patch_scrub_dbg == -1)) {
                DOUTR(true, "   trashing variable immediately after compute: "
                                << dws[dw]->getID() << "/" << patch->getID()
                                << "/" << matls->get(m) << "/"
                                << comp->m_var->getName());
              }
              dws[dw]->scrub(comp->m_var, matl, patch);
            }
          }
        }
      }
    }
  }
} // end scrub()

//_____________________________________________________________________________
//
void DetailedTask::findRequiringTasks(
    const VarLabel *var, std::list<DetailedTask *> &requiringTasks) {
  // find external requires
  for (DependencyBatch *batch = getComputes(); batch != nullptr;
       batch = batch->m_comp_next) {
    for (DetailedDep *dep = batch->m_head; dep != nullptr; dep = dep->m_next) {
      if (dep->m_req->m_var == var) {
        requiringTasks.insert(requiringTasks.end(), dep->m_to_tasks.begin(),
                              dep->m_to_tasks.end());
      }
    }
  }

  // find internal requires
  std::map<DetailedTask *, InternalDependency *>::iterator internalDepIter;
  for (internalDepIter = m_internal_dependents.begin();
       internalDepIter != m_internal_dependents.end(); ++internalDepIter) {
    if (internalDepIter->second->m_vars.find(var) !=
        internalDepIter->second->m_vars.end()) {
      requiringTasks.push_back(internalDepIter->first);
    }
  }
}

//_____________________________________________________________________________
//
void DetailedTask::addComputes(DependencyBatch *comp) {
  comp->m_comp_next = m_comp_head;
  m_comp_head = comp;
}

//_____________________________________________________________________________
//
bool DetailedTask::addRequires(DependencyBatch *req) {
  // return true if it is adding a new batch
  return m_reqs.insert(std::make_pair(req, req)).second;
}

//_____________________________________________________________________________
//
void DetailedTask::addInternalComputes(DependencyBatch *comp) {
  comp->m_comp_next = m_internal_comp_head;
  m_internal_comp_head = comp;
}

//_____________________________________________________________________________
//
bool DetailedTask::addInternalRequires(DependencyBatch *req) {
  // return true if it is adding a new batch
  return m_internal_reqs.insert(std::make_pair(req, req)).second;
}

//_____________________________________________________________________________
// can be called in one of two places - when the last MPI Recv has completed, or
// from MPIScheduler
void DetailedTask::checkExternalDepCount() {
  std::lock_guard<Uintah::MasterLock> external_ready_guard(
      g_external_ready_mutex);

  DOUTR(g_external_deps_dbg,
        "  DetailedTask::checkExternalDepCoun Task "
            << this->getTask()->getName() << " external deps: "
            << m_external_dependency_count.load(std::memory_order_acquire)
            << " internal deps: " << m_num_pending_internal_dependencies);

  if ((m_external_dependency_count.load(std::memory_order_acquire) == 0) &&
      m_task_group->m_sched_common->useInternalDeps() &&
      m_initiated.load(std::memory_order_acquire) && !m_task->usesMPI()) {

    DOUTR(g_external_deps_dbg, "    Task "
                                   << this->getTask()->getName()
                                   << " MPI requirements satisfied, placing "
                                      "into external ready queue");

    if (m_externally_ready.load(std::memory_order_acquire) == false) {
      m_task_group->m_mpi_completed_tasks.push(this);
      m_task_group->m_atomic_mpi_completed_tasks_size.fetch_add(1);
      m_externally_ready.store(true, std::memory_order_release);
    }
  }
}

//_____________________________________________________________________________
//
void DetailedTask::resetDependencyCounts() {
  m_external_dependency_count.store(0, std::memory_order_relaxed);
  m_externally_ready.store(false, std::memory_order_relaxed);
  m_initiated.store(false, std::memory_order_relaxed);

  m_wait_timer.reset(true);
  m_exec_timer.reset(true);
}

//_____________________________________________________________________________
//
void DetailedTask::addInternalDependency(DetailedTask *prerequisiteTask,
                                         const VarLabel *var) {
  if (m_task_group->mustConsiderInternalDependencies()) {
    // Avoid unnecessary multiple internal dependency links between tasks.
    std::map<DetailedTask *, InternalDependency *>::iterator foundIt =
        prerequisiteTask->m_internal_dependents.find(this);

    if (foundIt == prerequisiteTask->m_internal_dependents.end()) {
      m_internal_dependencies.push_back(InternalDependency(
          prerequisiteTask, this, var, 0 /* 0 == not satisfied */));
      prerequisiteTask->m_internal_dependents[this] =
          &m_internal_dependencies.back();
      m_num_pending_internal_dependencies = m_internal_dependencies.size();

      DOUTR(g_internal_deps_dbg,
            "  DetailedTask::addInternalDependency  Adding dependency between "
                << *this << " and " << *prerequisiteTask << " for var "
                << var->getName() << " source dep count: "
                << m_num_pending_internal_dependencies << " pre-req dep count "
                << prerequisiteTask->m_internal_dependents.size());
    } else {
      foundIt->second->addVarLabel(var);
    }
  }
}

//_____________________________________________________________________________
//
void DetailedTask::done(std::vector<OnDemandDataWarehouseP> &dws) {
  // Important to scrub first, before dealing with the internal dependencies
  scrub(dws);

  if (g_internal_deps_dbg) {
    std::ostringstream message;
    DOUTR(true, "  DetailedTask::done  This: "
                    << this->getTask()->getName()
                    << " is done with task: " << m_task->getName()
                    << "  which has (" << m_internal_dependents.size()
                    << ") tasks waiting on it:");
  }

  for (auto iter = m_internal_dependents.begin();
       iter != m_internal_dependents.end(); ++iter) {
    InternalDependency *dep = (*iter).second;
    dep->m_dependent_task->dependencySatisfied(dep);

    DOUTR(g_internal_deps_dbg, "    Dependency satisfied between "
                                   << *dep->m_dependent_task << " and "
                                   << *this);
  }

  m_exec_timer.stop();
}

//_____________________________________________________________________________
//
void DetailedTask::dependencySatisfied(InternalDependency *dep) {
  std::lock_guard<Uintah::MasterLock> internal_dependency_guard(
      g_internal_dependency_mutex);

  ASSERT(m_num_pending_internal_dependencies > 0);
  unsigned long currentGeneration =
      m_task_group->getCurrentDependencyGeneration();

  // if false, then the dependency has already been satisfied
  ASSERT(dep->m_satisfied_generation < currentGeneration);

  dep->m_satisfied_generation = currentGeneration;
  m_num_pending_internal_dependencies--;

  DOUTR(g_internal_deps_dbg,
        "  DetailedTask::dependencySatisfied"
            << *(dep->m_dependent_task->getTask()) << " has "
            << m_num_pending_internal_dependencies << " left.");

  DOUTR(g_internal_deps_dbg,
        "   satisfying dependency: prereq: "
            << *dep->m_prerequisite_task << " dep: " << *dep->m_dependent_task
            << " numPending: " << m_num_pending_internal_dependencies);

  if (m_num_pending_internal_dependencies == 0) {
    m_task_group->internalDependenciesSatisfied(this);
    m_num_pending_internal_dependencies =
        m_internal_dependencies.size(); // reset for next timestep
  }
}

//_____________________________________________________________________________
//
void DetailedTask::emitEdges(ProblemSpecP edgesElement) {
  for (auto req_iter = m_reqs.begin(); req_iter != m_reqs.end(); ++req_iter) {
    DetailedTask *fromTask = (*req_iter).first->m_from_task;
    ProblemSpecP edge = edgesElement->appendChild("edge");
    edge->appendElement("source", fromTask->getName());
    edge->appendElement("target", getName());
  }

  for (auto iter = m_internal_dependencies.begin();
       iter != m_internal_dependencies.end(); ++iter) {
    DetailedTask *fromTask = (*iter).m_prerequisite_task;
    if (getTask()->isReductionTask() &&
        fromTask->getTask()->isReductionTask()) {
      // Ignore internal links between reduction tasks because they
      // are only needed for logistic reasons
      continue;
    }
    ProblemSpecP edge = edgesElement->appendChild("edge");
    edge->appendElement("source", fromTask->getName());
    edge->appendElement("target", getName());
  }
}

//_____________________________________________________________________________
//
class PatchIDIterator {

public:
  PatchIDIterator(const std::vector<const Patch *>::const_iterator &iter)
      : m_const_iter(iter) {}

  PatchIDIterator &operator=(const PatchIDIterator &iter2) {
    m_const_iter = iter2.m_const_iter;
    return *this;
  }

  int operator*() {
    const Patch *patch =
        *m_const_iter; // vector<Patch*>::iterator::operator*();
    return patch ? patch->getID() : -1;
  }

  PatchIDIterator &operator++() {
    m_const_iter++;
    return *this;
  }

  bool operator!=(const PatchIDIterator &iter2) {
    return m_const_iter != iter2.m_const_iter;
  }

private:
  std::vector<const Patch *>::const_iterator m_const_iter;
};

//_____________________________________________________________________________
//
std::string DetailedTask::getName() const {
  if (m_name != "") {
    return m_name;
  }

  m_name = std::string(m_task->getName());

  if (m_patches != nullptr) {
    ConsecutiveRangeSet patchIDs;
    patchIDs.addInOrder(PatchIDIterator(m_patches->getVector().begin()),
                        PatchIDIterator(m_patches->getVector().end()));
    m_name += std::string(" (Patches: ") + patchIDs.toString() + ")";
  }

  if (m_matls != nullptr) {
    ConsecutiveRangeSet matlSet;
    matlSet.addInOrder(m_matls->getVector().begin(),
                       m_matls->getVector().end());
    m_name += std::string(" (Matls: ") + matlSet.toString() + ")";
  }

  return m_name;
}

namespace Uintah {

std::ostream &operator<<(std::ostream &out, const DetailedTask &dtask) {
  g_dtask_output_mutex.lock();
  {
    out << dtask.getTask()->getName();
    const PatchSubset *patches = dtask.getPatches();
    if (patches) {

      out << ", on patch";
      if (patches->size() > 1) {
        out << "es";
      }
      out << " ";
      for (int i = 0; i < patches->size(); i++) {
        if (i > 0) {
          out << ",";
        }
        out << patches->get(i)->getID();
      }
      // a once-per-proc task is liable to have multiple levels, and thus calls
      // to getLevel(patches) will fail
      if (dtask.getTask()->getType() == Task::TaskType::OncePerProc ||
          dtask.getTask()->getType() == Task::TaskType::Hypre) {
        out << ", on multiple levels";
      } else {
        out << ", Level " << getLevel(patches)->getIndex();
      }
    }
    const MaterialSubset *matls = dtask.getMaterials();
    if (matls) {
      out << ", on material";
      if (matls->size() > 1) {
        out << "s";
      }
      out << " ";
      for (int i = 0; i < matls->size(); i++) {
        if (i > 0) {
          out << ",";
        }
        out << matls->get(i);
      }
    }
    out << ", resource (rank): ";
    if (dtask.getAssignedResourceIndex() == -1) {
      out << "unassigned";
    } else {
      out << dtask.getAssignedResourceIndex();
    }
  }
  g_dtask_output_mutex.unlock();

  return out;
}

} // end namespace Uintah

#if defined(UINTAH_ENABLE_DEVICE)

// For tasks where there are multiple devices & its corresponding stream for the task (i.e. data archiver
// output tasks)
std::set<int> DetailedTask::getDeviceNums() const { return deviceNums_; }

gpuStream_t* DetailedTask::getGpuStreamForThisTask(int device_id) const {
  auto it = d_gpuStreams.find(device_id);
  if (it != d_gpuStreams.end()) {
    return it->second;
  } else {
    SCI_THROW(InternalError(
        "A GPU stream not assigned to the task: " + getName(),
        __FILE__, __LINE__));
    exit(-1);
  }
}

void DetailedTask::setGpuStreamForThisTask(int device_id) {
  // ABB: This function gets called from UnifiedScheduler multiple times
  // if the device_id already exists and a valid stream is assigned skip this and return
  if (deviceNums_.find(device_id) == deviceNums_.end()) {
    deviceNums_.insert(device_id);
    d_gpuStreams.try_emplace(device_id, GPUStreamPool<>::getInstance().getGpuStreamFromPool(device_id));
  }
  return;
};

void DetailedTask::clearGpuStreamsForThisTask() {
  d_gpuStreams.clear();
}

//_____________________________________________________________________________
//

bool DetailedTask::checkAllGpuStreamsDoneForThisTask() const {
  // A task can have multiple devices/streams (such as an output task pulling from
  // multiple GPUs). Check all streams to see if they are done. If any one
  // stream isn't done, return false.  If nothing returned true, then they all
  // must be good to go.

  for (auto it = d_gpuStreams.cbegin(); it != d_gpuStreams.cend(); ++it) {
    gpuSetDevice(it->first);
    gpuError_t retVal = gpuStreamQuery(*(it->second));
    if (retVal != gpuSuccess) {
      return false;
    }
  }
  return true;
}

//_____________________________________________________________________________
//
void DetailedTask::setTaskGpuDataWarehouse(const int whichDevice,
                                           Task::WhichDW DW,
                                           GPUDataWarehouse *TaskDW) {

  auto iter = TaskGpuDWs.find(whichDevice);
  if (iter != TaskGpuDWs.end()) {
    iter->second.TaskGpuDW[static_cast<int>(DW)] = TaskDW;
  } else {
    TaskGpuDataWarehouses temp;
    temp.TaskGpuDW[0] = nullptr;
    temp.TaskGpuDW[1] = nullptr;
    temp.TaskGpuDW[static_cast<int>(DW)] = TaskDW;
    TaskGpuDWs.try_emplace(whichDevice, temp);
  }
}

//_____________________________________________________________________________
//
GPUDataWarehouse *DetailedTask::getTaskGpuDataWarehouse(const int whichDevice,
                                                        Task::WhichDW DW) {
  auto iter = TaskGpuDWs.find(whichDevice);
  if (iter != TaskGpuDWs.end()) {
    return iter->second.TaskGpuDW[static_cast<int>(DW)];
  }
  return nullptr;
}

void DetailedTask::deleteTaskGpuDataWarehouses() {
  for (auto iter = TaskGpuDWs.begin(); iter != TaskGpuDWs.end(); ++iter) {
    for (int i = 0; i < 2; i++) {
      if (iter->second.TaskGpuDW[i] != nullptr) {
        // Note: Do not call the clear() method.  The Task GPU DWs only contains
        // a "snapshot" of the things in the GPU.  The host side GPU DWs is
        // responsible for deallocating all the GPU resources.  The only thing
        // we do want to clean up is that this GPUDW lives on the GPU.
        iter->second.TaskGpuDW[i]->deleteSelfOnDevice();
        iter->second.TaskGpuDW[i]->cleanup();

        free(iter->second.TaskGpuDW[i]);
        iter->second.TaskGpuDW[i] = nullptr;
      }
    }
  }
}

void DetailedTask::clearPreparationCollections() {
  deviceVars.clear();
  ghostVars.clear();
  taskVars.clear();
  varsToBeGhostReady.clear();
  varsBeingCopiedByTask.clear();
}

void DetailedTask::addTempHostMemoryToBeFreedOnCompletion(void *host_ptr) {
  taskHostMemoryPoolItems.push(host_ptr);
}
void *
DetailedTask::addTempGpuMemoryToBeFreedOnCompletion(unsigned int device_id,
                                                    std::size_t sizeinbytes) {
  void *dev_ptr = GPUMemoryPool::getInstance().allocate(device_id, sizeinbytes);
  gpuMemoryPoolDevicePtrItem gpuItem(device_id, dev_ptr, sizeinbytes);
  taskGpuMemoryPoolItems.push_back(gpuItem);

  return dev_ptr;
}

void DetailedTask::deleteTemporaryTaskVars() {
  // clean out the host list
  while (!taskHostMemoryPoolItems.empty()) {
    std::cout << "1. issue from here ?? \n";
    ::operator delete(taskHostMemoryPoolItems.front());
    std::cout << "2. issue from here ?? \n";
    // TODO: Deletes a void*, and that doesn't call any object destructors
    // delete[] taskHostMemoryPoolItems.front();
    taskHostMemoryPoolItems.pop();
  }

  // and the device
  auto &memGPUPool = GPUMemoryPool::getInstance();
  for (auto p : taskGpuMemoryPoolItems) {
    memGPUPool.deallocate(p.device_id, p.ptr, p.sizeinbytes);
  }
  taskGpuMemoryPoolItems.clear();
}

#endif //UINTAH_ENABLE_DEVICE
