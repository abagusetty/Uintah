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

#include <CCA/Components/Schedulers/OnDemandDataWarehouse.h>
#include <CCA/Components/Schedulers/SYCLScheduler.hpp>
#include <CCA/Components/Schedulers/TaskGraph.h>
#include <CCA/Ports/Output.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Parallel/CommunicationList.hpp>
#include <Core/Parallel/MasterLock.h>

#include <Core/Grid/Variables/GPUGridVariable.h>
#include <Core/Grid/Variables/GPUStencil7.h>

#include <atomic>
#include <cstring>
#include <iomanip>
#include <thread>

using namespace Uintah;

//______________________________________________________________________
//
namespace {
Uintah::MasterLock g_scheduler_mutex{}; // main scheduler lock for
                                        // multi-threaded task selection
Uintah::MasterLock
    g_mark_task_consumed_mutex{}; // allow only one task at a time to enter the
                                  // task consumed section
Uintah::MasterLock g_lb_mutex{};  // load balancer lock
} // namespace

namespace {
Uintah::MasterLock
    g_GridVarSuperPatch_mutex{}; // An ugly hack to get superpatches for host
                                 // levels to work.
}

//______________________________________________________________________
//
namespace Uintah {

using lock_guard = std::lock_guard<Uintah::MasterLock>;

namespace Impl {

namespace {
thread_local int t_tid = 0; // unique ID assigned in thread_driver()
}

namespace {

enum class ThreadState : int { Inactive, Active, Exit };

SYCLSchedulerWorker *g_runners[MAX_THREADS] = {};
std::vector<std::thread> g_threads{};
volatile ThreadState g_thread_states[MAX_THREADS] = {};
int g_cpu_affinities[MAX_THREADS] = {};
int g_num_threads = 0;

std::atomic<int> g_run_tasks{0};

//______________________________________________________________________
//
void set_affinity(const int proc_unit) {
  cpu_set_t mask;
  unsigned int len = sizeof(mask);
  CPU_ZERO(&mask);
  CPU_SET(proc_unit, &mask);
  sched_setaffinity(0, len, &mask);
}

//______________________________________________________________________
//
void thread_driver(const int tid) {
  // t_tid is a thread_local variable, unique to each std::thread spawned below
  t_tid = tid;

  // set each TaskWorker thread's affinity
  set_affinity(g_cpu_affinities[tid]);

  try {
    // wait until main thread sets function and changes states
    g_thread_states[tid] = ThreadState::Inactive;
    while (g_thread_states[tid] == ThreadState::Inactive) {
      std::this_thread::yield();
    }

    while (g_thread_states[tid] == ThreadState::Active) {

      // run the function and wait for main thread to reset state
      g_runners[tid]->run();

      g_thread_states[tid] = ThreadState::Inactive;
      while (g_thread_states[tid] == ThreadState::Inactive) {
        std::this_thread::yield();
      }
    }

  } catch (const std::exception &e) {
    std::cerr << "Exception thrown from worker thread: " << e.what()
              << std::endl;
    std::cerr.flush();
    std::abort();
  } catch (...) {
    std::cerr << "Unknown Exception thrown from worker thread" << std::endl;
    std::cerr.flush();
    std::abort();
  }
}

//______________________________________________________________________
// only called by thread 0 (main thread)
void thread_fence() {
  // main thread tid is at [0]
  g_thread_states[0] = ThreadState::Inactive;

  // TaskRunner threads start at [1]
  for (int i = 1; i < g_num_threads; ++i) {
    while (g_thread_states[i] == ThreadState::Active) {
      std::this_thread::yield();
    }
  }
  std::atomic_thread_fence(std::memory_order_seq_cst);
}

//______________________________________________________________________
// only called by main thread
void init_threads(SYCLScheduler *sched, int num_threads) {
  // we now need to refer to the total number of active threads (set in
  // Uintah::Parallel, i.e., num_threads + 1)
  g_num_threads = num_threads + 1;

  for (int i = 0; i < g_num_threads; ++i) {
    g_thread_states[i] = ThreadState::Active;
    g_cpu_affinities[i] = i;
  }

  // set main thread's affinity and tid, core-0 and tid-0, respectively
  set_affinity(g_cpu_affinities[0]);
  t_tid = 0;

  // TaskRunner threads start at g_runners[1], and std::threads start at
  // g_threads[1]
  for (int i = 1; i < g_num_threads; ++i) {
    g_runners[i] = new SYCLSchedulerWorker(sched, i, g_cpu_affinities[i]);
    Impl::g_threads.push_back(std::thread(thread_driver, i));
  }

  for (auto &t : Impl::g_threads) {
    if (t.joinable()) {
      t.detach();
    }
  }

  thread_fence();
}

} // namespace
} // namespace Impl
} // namespace Uintah

//______________________________________________________________________
//
SYCLScheduler::SYCLScheduler(const ProcessorGroup *myworld,
                             SYCLScheduler *parentScheduler)
    : MPIScheduler(myworld, parentScheduler) {

  if (Uintah::Parallel::usingDevice()) {
    gpuInitialize();

    // disable memory windowing on variables.  This will ensure that
    // each variable is allocated its own memory on each patch,
    // precluding memory blocks being defined across multiple patches.
    Uintah::OnDemandDataWarehouse::s_combine_memory = false;

    auto const &gpu_devices =
        sycl::device::get_devices(sycl::info::device_type::gpu);
    std::vector<ze_device_handle_t> ze_devices{};
    for (int i = 0; i < gpu_devices.size(); i++) {
      if (gpu_devices[i]
              .get_info<sycl::info::device::partition_max_sub_devices>() > 0) {
        auto SubDevices =
            gpu_devices[i]
                .create_sub_devices<sycl::info::partition_property::
                                        partition_by_affinity_domain>(
                    sycl::info::partition_affinity_domain::numa);
        for (const auto &tile : SubDevices) {
          ze_devices.push_back(
              sycl::get_native<sycl::backend::ext_oneapi_level_zero>(tile));
        }
      }
    }

    int num_devices = ze_devices.size();
    ze_bool_t can_access = false;
    for (int i = 0; i < num_devices; i++) {
      for (int j = 0; j < num_devices; j++) {
        if (i != j) {
          zeDeviceCanAccessPeer(ze_devices[i], ze_devices[j], &can_access);
          if (!can_access) {
            std::cout << "ERROR\n GPU device [" << i
                      << "] cannot peer access GPU device [" << j << "] "
                      << std::endl;
            SCI_THROW(InternalError("** ERROR P2P peer GPU not supported.",
                                    __FILE__, __LINE__));
          }
        }
      }
    }
  } // using Device
}

SYCLScheduler::~SYCLScheduler() {}

void SYCLScheduler::problemSetup(const ProblemSpecP &prob_spec,
                                 const MaterialManagerP &materialManager) {
  // Default taskReadyQueueAlg
  std::string taskQueueAlg = "";

  ProblemSpecP params = prob_spec->findBlock("Scheduler");
  if (params) {
    params->get("taskReadyQueueAlg", taskQueueAlg);
    if (taskQueueAlg == "") {
      taskQueueAlg = "MostMessages"; // default taskReadyQueueAlg
    }
    if (taskQueueAlg == "FCFS") {
      m_task_queue_alg = FCFS;
    } else if (taskQueueAlg == "Stack") {
      m_task_queue_alg = Stack;
    } else if (taskQueueAlg == "Random") {
      m_task_queue_alg = Random;
    } else if (taskQueueAlg == "MostChildren") {
      m_task_queue_alg = MostChildren;
    } else if (taskQueueAlg == "LeastChildren") {
      m_task_queue_alg = LeastChildren;
    } else if (taskQueueAlg == "MostAllChildren") {
      m_task_queue_alg = MostAllChildren;
    } else if (taskQueueAlg == "LeastAllChildren") {
      m_task_queue_alg = LeastAllChildren;
    } else if (taskQueueAlg == "MostL2Children") {
      m_task_queue_alg = MostL2Children;
    } else if (taskQueueAlg == "LeastL2Children") {
      m_task_queue_alg = LeastL2Children;
    } else if (taskQueueAlg == "MostMessages") {
      m_task_queue_alg = MostMessages;
    } else if (taskQueueAlg == "LeastMessages") {
      m_task_queue_alg = LeastMessages;
    } else if (taskQueueAlg == "PatchOrder") {
      m_task_queue_alg = PatchOrder;
    } else if (taskQueueAlg == "PatchOrderRandom") {
      m_task_queue_alg = PatchOrderRandom;
    } else {
      throw ProblemSetupException("Unknown task ready queue algorithm",
                                  __FILE__, __LINE__);
    }
  }

  proc0cout << "Using \"" << taskQueueAlg << "\" task queue priority algorithm"
            << std::endl;

  int num_threads = Uintah::Parallel::getNumThreads() - 1;

  if ((num_threads < 1) && Uintah::Parallel::usingDevice()) {
    if (d_myworld->myRank() == 0) {
      std::cerr << "Error: no thread number specified for SYCL Scheduler"
                << std::endl;
      throw ProblemSetupException(
          "This scheduler requires number of threads to be in the range [2, "
          "64],\n.... please use -nthreads <num>, and -gpu if using GPUs",
          __FILE__, __LINE__);
    }
  } else if (num_threads > MAX_THREADS) {
    if (d_myworld->myRank() == 0) {
      std::cerr << "Error: Number of threads too large..." << std::endl;
      throw ProblemSetupException(
          "Too many threads. Reduce MAX_THREADS and recompile.", __FILE__,
          __LINE__);
    }
  }

  if (d_myworld->myRank() == 0) {
    std::string plural = (num_threads == 1) ? " thread" : " threads";
    std::cout << "\nWARNING: Multi-threaded SYCL scheduler is EXPERIMENTAL, "
                 "not all tasks are thread safe yet.\n"
              << "Creating " << num_threads << " additional "
              << plural + " for task execution (total task execution threads = "
              << num_threads + 1 << ").\n"
              << std::endl;

    if (Uintah::Parallel::usingDevice()) {
      int availableDevices = 0;
      syclGetDeviceCount(&availableDevices);
      std::cout << "   Using " << m_num_devices << "/" << availableDevices
                << " available GPU(s)" << std::endl;

      int tmp_device_id = 0;
      auto const &gpu_devices =
          sycl::device::get_devices(sycl::info::device_type::gpu);
      for (int device_id = 0; device_id < gpu_devices.size(); device_id++) {
        if (gpu_devices[device_id]
                .get_info<sycl::info::device::partition_max_sub_devices>() >
            0) {
          auto subDevices =
              gpu_devices[device_id]
                  .create_sub_devices<sycl::info::partition_property::
                                          partition_by_affinity_domain>(
                      sycl::info::partition_affinity_domain::numa);
          for (const auto &tile : subDevices) {
            std::cout << "   GPU Device " << tmp_device_id << " "
                      << tile.get_info<sycl::info::device::name>()
                      << ": with compute capability [backend: "
                      << tile.get_backend() << "] v:"
                      << tile.get_info<sycl::info::device::driver_version>()
                      << std::endl;
            tmp_device_id++;
          }
        } else {
          std::cout
              << "   GPU Device " << tmp_device_id << " "
              << gpu_devices[device_id].get_info<sycl::info::device::name>()
              << ": with compute capability [backend: "
              << gpu_devices[device_id].get_backend() << "] v:"
              << gpu_devices[device_id]
                     .get_info<sycl::info::device::driver_version>()
              << std::endl;
          tmp_device_id++;
        }
      }
    }
  }

  if (Uintah::Parallel::usingDevice()) {
    int availableDevices = 0;
    std::ostringstream message;
    syclGetDeviceCount(&availableDevices);
    message << "   Rank-" << d_myworld->myRank() << " using " << m_num_devices
            << "/" << availableDevices << " available GPU(s)\n";
  }

  SchedulerCommon::problemSetup(prob_spec, materialManager);

  // Now pick out the materials out of the file.  This is done with an
  // assumption that there will only be ICE or MPM problems, and no problem will
  // have both ICE and MPM materials in it. I am unsure if this assumption is
  // correct.
  // TODO: Add in MPM material support, just needs to look for an MPM block
  // instead of an ICE block.
  ProblemSpecP mp = prob_spec->findBlockWithOutAttribute("MaterialProperties");
  if (mp) {
    ProblemSpecP group = mp->findBlock("ICE");
    if (group) {
      for (ProblemSpecP child = group->findBlock("material"); child != nullptr;
           child = child->findNextBlock("material")) {
        ProblemSpecP EOS_ps = child->findBlock("EOS");
        if (!EOS_ps) {
          throw ProblemSetupException("ERROR ICE: Cannot find EOS tag",
                                      __FILE__, __LINE__);
        }

        std::string EOS;
        if (!EOS_ps->getAttribute("type", EOS)) {
          throw ProblemSetupException("ERROR ICE: Cannot find EOS 'type' tag",
                                      __FILE__, __LINE__);
        }

        // add this material to the collection of materials
        m_material_names.push_back(EOS);
      }
    }
  }

  // this spawns threads, sets affinity, etc
  init_threads(this, num_threads);
}

//______________________________________________________________________
//
SchedulerP SYCLScheduler::createSubScheduler() {
  return MPIScheduler::createSubScheduler();
}

//______________________________________________________________________
//
void SYCLScheduler::runTask(DetailedTask *dtask, int iteration,
                            int thread_id, Task::CallBackEvent event) {
  std::cout << "US::runTask() " << dtask->getName() << ", " << iteration << ", "
            << thread_id << ", " << event << std::endl;

  // Only execute CPU or GPU tasks.  Don't execute postGPU tasks a second time.
  if (event == Task::CPU || event == Task::GPU) {
    std::vector<DataWarehouseP> plain_old_dws(m_dws.size());
    for (int i = 0; i < static_cast<int>(m_dws.size()); i++) {
      plain_old_dws[i] = m_dws[i].get_rep();
    }

    dtask->doit(d_myworld, m_dws, plain_old_dws, event);
  }

  // For CPU and postGPU task runs, post MPI sends and call task->done;
  if (event == Task::CPU || event == Task::postGPU) {

    if (Uintah::Parallel::usingDevice()) {

      // TODO: Don't make every task run through this
      // TODO: Verify that it only looks for data that's valid in the GPU, and
      // not assuming it's valid.
      // Load up the prepareDeviceVars by preparing ghost cell regions to copy
      // out.
      findIntAndExtGpuDependencies(dtask, iteration, thread_id);

      // The ghost cell destinations indicate which devices we're using,
      // and which ones we'll need streams for.
      assignDevicesAndStreamsFromGhostVars(dtask);
      createTaskGpuDWs(dtask);

      // place everything in the GPU data warehouses
      prepareDeviceVars(dtask);
      prepareTaskVarsIntoTaskDW(dtask);
      prepareGhostCellsIntoTaskDW(dtask);
      syncTaskGpuDWs(dtask);

      // get these ghost cells to contiguous arrays so they can be copied to
      // host.
      performInternalGhostCellCopies(dtask); // TODO: Fix for multiple GPUs

      // Now that we've done internal ghost cell copies, we can mark the staging
      // vars as being valid.
      // TODO: Sync required?  We shouldn't mark data as valid until it has
      // copied.
      markDeviceRequiresDataAsValid(dtask);

      copyAllGpuToGpuDependences(dtask);
      // TODO: Mark destination staging vars as valid.

      // copy all dependencies to arrays
      copyAllExtGpuDependenciesToHost(dtask);

      // In order to help copy values to another on-node GPU or another MPI
      // rank, ghost cell data was placed in a var in the patch it is *going
      // to*.  It helps reuse gpu dw engine code this way. But soon, after this
      // task is done, we are likely going to receive a different region of that
      // patch from a neighboring on-node GPU or neighboring MPI rank.  So we
      // need to remove this foreign variable now so it can be used again.
      // clearForeignGpuVars(deviceVars);
    }

    MPIScheduler::postMPISends(dtask, iteration);

    if (Uintah::Parallel::usingDevice()) {
      dtask->deleteTaskGpuDataWarehouses();
    }

    dtask->done(m_dws);

    lock_guard load_bal_guard(g_lb_mutex);
    {
      // Do the global and local per task monitoring
      sumTaskMonitoringValues(dtask);

      double total_task_time = dtask->task_exec_time();

      // if I do not have a sub scheduler
      if (!dtask->getTask()->getHasSubScheduler()) {
        // add my task time to the total time
        m_mpi_info[TotalTask] += total_task_time;
        if (!m_is_copy_data_timestep &&
            dtask->getTask()->getType() != Task::Output) {
          // add contribution for patchlist
          m_loadBalancer->addContribution(dtask, total_task_time);
        }
      }
    }

    //---------------------------------------------------------------------------
    // New way of managing single MPI requests - avoids MPI_Waitsome &
    // MPI_Donesome - APH 07/20/16
    //---------------------------------------------------------------------------
    // test a pending request
    auto ready_request = [](CommRequest const &r) -> bool { return r.test(); };
    CommRequestPool::handle find_handle;
    CommRequestPool::iterator comm_sends_iter =
        m_sends.find_any(find_handle, ready_request);
    if (comm_sends_iter) {
      MPI_Status status;
      comm_sends_iter->finishedCommunication(d_myworld, status);
      find_handle = comm_sends_iter;
      m_sends.erase(comm_sends_iter);
    }
    //---------------------------------------------------------------------------

    // Add subscheduler timings to the parent scheduler and reset subscheduler
    // timings
    if (m_parent_scheduler != nullptr) {
      for (std::size_t i = 0; i < m_mpi_info.size(); ++i) {
        m_parent_scheduler->m_mpi_info[i] += m_mpi_info[i];
      }
      m_mpi_info.reset(0);
      m_thread_info.reset(0);
    }
  }
} // end runTask()

//______________________________________________________________________
//

void SYCLScheduler::execute(int tgnum, int iteration) {

  // copy data timestep must be single threaded for now and
  //  also needs to run deterministically, in a static order
  if (m_is_copy_data_timestep) {
    MPIScheduler::execute(tgnum, iteration);
    return;
  }

  ASSERTRANGE(tgnum, 0, static_cast<int>(m_task_graphs.size()));
  TaskGraph *tg = m_task_graphs[tgnum];
  tg->setIteration(iteration);
  m_current_task_graph = tgnum;

  if (m_task_graphs.size() > 1) {
    // tg model is the multi TG model, where each graph is going to need to
    // have its dwmap reset here (even with the same tgnum)
    tg->remapTaskDWs(m_dwmap);
  }

  m_detailed_tasks = tg->getDetailedTasks();
  if (m_detailed_tasks == nullptr) {
    proc0cout << "SYCLScheduler skipping execute, no tasks\n";
    return;
  }

  m_detailed_tasks->initializeScrubs(m_dws, m_dwmap);
  m_detailed_tasks->initTimestep();

  m_num_tasks = m_detailed_tasks->numLocalTasks();

  for (int i = 0; i < m_num_tasks; i++) {
    m_detailed_tasks->localTask(i)->resetDependencyCounts();
  }

  m_mpi_info.reset(0);
  m_thread_info.reset(0);

  m_num_tasks_done = 0;
  m_abort = false;
  m_abort_point = 987654;

  if (m_reloc_new_pos_label && m_dws[m_dwmap[Task::OldDW]] != nullptr) {
    m_dws[m_dwmap[Task::OldDW]]->exchangeParticleQuantities(
        m_detailed_tasks, m_loadBalancer, m_reloc_new_pos_label, iteration);
  }

  m_curr_iteration = iteration;
  m_curr_phase = 0;
  m_num_phases = tg->getNumTaskPhases();
  m_phase_tasks.clear();
  m_phase_tasks.resize(m_num_phases, 0);
  m_phase_tasks_done.clear();
  m_phase_tasks_done.resize(m_num_phases, 0);
  m_phase_sync_task.clear();
  m_phase_sync_task.resize(m_num_phases, nullptr);
  m_detailed_tasks->setTaskPriorityAlg(m_task_queue_alg);

  // get the number of tasks in each task phase
  for (int i = 0; i < m_num_tasks; i++) {
    m_phase_tasks[m_detailed_tasks->localTask(i)->getTask()->m_phase]++;
  }

  //------------------------------------------------------------------------------------------------
  // activate TaskRunners
  //------------------------------------------------------------------------------------------------
  if (!m_is_copy_data_timestep) {
    Impl::g_run_tasks.store(1, std::memory_order_relaxed);
    for (int i = 1; i < Impl::g_num_threads; ++i) {
      Impl::g_thread_states[i] = Impl::ThreadState::Active;
    }
  }
  //------------------------------------------------------------------------------------------------

  // main thread also executes tasks
  runTasks(Impl::t_tid);

  //------------------------------------------------------------------------------------------------
  // deactivate TaskRunners
  //------------------------------------------------------------------------------------------------
  if (!m_is_copy_data_timestep) {
    Impl::g_run_tasks.store(0, std::memory_order_relaxed);

    Impl::thread_fence();

    for (int i = 1; i < Impl::g_num_threads; ++i) {
      Impl::g_thread_states[i] = Impl::ThreadState::Inactive;
    }
  }
  //------------------------------------------------------------------------------------------------

  //---------------------------------------------------------------------------
  // New way of managing single MPI requests - avoids MPI_Waitsome &
  // MPI_Donesome - APH 07/20/16
  //---------------------------------------------------------------------------
  // wait on all pending requests
  auto ready_request = [](CommRequest const &r) -> bool { return r.wait(); };
  CommRequestPool::handle find_handle;
  while (m_sends.size() != 0u) {
    CommRequestPool::iterator comm_sends_iter;
    if ((comm_sends_iter = m_sends.find_any(find_handle, ready_request))) {
      find_handle = comm_sends_iter;
      m_sends.erase(comm_sends_iter);
    } else {
      // TODO - make this a sleep? APH 07/20/16
    }
  }
  //---------------------------------------------------------------------------

  ASSERT(m_sends.size() == 0u);
  ASSERT(m_recvs.size() == 0u);

  finalizeTimestep();
}

//______________________________________________________________________
//
void SYCLScheduler::markTaskConsumed(int &numTasksDone, int &currphase,
                                     int numPhases, DetailedTask *dtask) {
  std::lock_guard<Uintah::MasterLock> task_consumed_guard(
      g_mark_task_consumed_mutex);

  // Update the count of tasks consumed by the scheduler.
  numTasksDone++;

  // task ordering debug info - please keep this here, APH 05/30/18

  // Update the count of this phase consumed.
  m_phase_tasks_done[dtask->getTask()->m_phase]++;

  // See if we've consumed all tasks on this phase, if so, go to the next phase.
  while (m_phase_tasks[currphase] == m_phase_tasks_done[currphase] &&
         currphase + 1 < numPhases) {
    currphase++;
  }
}

//______________________________________________________________________
//
void SYCLScheduler::runTasks(int thread_id) {

  while (m_num_tasks_done < m_num_tasks) {

    DetailedTask *readyTask = nullptr;
    DetailedTask *initTask = nullptr;

    bool havework = false;

    bool usingDevice = Uintah::Parallel::usingDevice();
    bool gpuInitReady = false;
    bool gpuValidateRequiresCopies = false;
    bool gpuPerformGhostCopies = false;
    bool gpuValidateGhostCopies = false;
    bool gpuCheckIfExecutable = false;
    bool gpuRunReady = false;
    bool gpuPending = false;
    bool cpuInitReady = false;
    bool cpuValidateRequiresCopies = false;
    bool cpuCheckIfExecutable = false;
    bool cpuRunReady = false;

    // ----------------------------------------------------------------------------------
    // Part 1:
    //    Check if anything this thread can do concurrently.
    //    If so, then update the various scheduler counters.
    // ----------------------------------------------------------------------------------
    // g_scheduler_mutex.lock();
    while (!havework) {
      /*
       * (1.1)
       *
       * If it is time to setup for a reduction task, then do so.
       *
       */

      if ((m_phase_sync_task[m_curr_phase] != nullptr) &&
          (m_phase_tasks_done[m_curr_phase] ==
           m_phase_tasks[m_curr_phase] - 1)) {
        g_scheduler_mutex.lock();
        if ((m_phase_sync_task[m_curr_phase] != nullptr) &&
            (m_phase_tasks_done[m_curr_phase] ==
             m_phase_tasks[m_curr_phase] - 1)) {
          readyTask = m_phase_sync_task[m_curr_phase];
          havework = true;
          markTaskConsumed(m_num_tasks_done, m_curr_phase, m_num_phases,
                           readyTask);
          cpuRunReady = true;
        }
        g_scheduler_mutex.unlock();
        break;
      }

      /*
       * (1.2)
       *
       * Run a CPU task that has its MPI communication complete. These tasks get
       * in the external ready queue automatically when their receive count hits
       * 0 in DependencyBatch::received, which is called when a MPI message is
       * delivered.
       *
       * NOTE: This is also where a GPU-enabled task gets into the GPU
       * initially-ready queue
       *
       */
      else if ((readyTask = m_detailed_tasks->getNextExternalReadyTask())) {
        havework = true;
        /*
         * (1.2.1)
         *
         * If it's a GPU-enabled task, assign it to a device (patches were
         * assigned devices previously) and initiate its H2D computes and
         * requires data copies. This is where the execution cycle begins for
         * each GPU-enabled Task.
         *
         * gpuInitReady = true
         */
        if (usingDevice == false || readyTask->getPatches() == nullptr) {
          // These tasks won't ever have anything to pull out of the device
          // so go ahead and mark the task "done" and say that it's ready
          // to start running as a CPU task.

          markTaskConsumed(m_num_tasks_done, m_curr_phase, m_num_phases,
                           readyTask);
          cpuRunReady = true;
        } else if (!readyTask->getTask()->usesDevice() && usingDevice) {
          // These tasks can't start unless we copy and/or verify all data into
          // host memory
          cpuInitReady = true;
        } else if (readyTask->getTask()->usesDevice()) {
          // These tasks can't start until we copy and/or verify all data into
          // GPU memory
          gpuInitReady = true;
        } else {
          markTaskConsumed(m_num_tasks_done, m_curr_phase, m_num_phases,
                           readyTask);
          cpuRunReady = true;
        }

        // if NOT compiled with device support, then this is a CPU task and we
        // can mark the task consumed
        markTaskConsumed(m_num_tasks_done, m_curr_phase, m_num_phases,
                         readyTask);
        break;
      }

      /*
       * (1.3)
       *
       * If we have an internally-ready CPU task, initiate its MPI receives,
       * preparing it for CPU external ready queue. The task is moved to the CPU
       * external-ready queue in the call to task->checkExternalDepCount().
       *
       */
      else if ((initTask = m_detailed_tasks->getNextInternalReadyTask())) {
        if (initTask->getTask()->getType() == Task::Reduction ||
            initTask->getTask()->usesMPI()) {
          m_phase_sync_task[initTask->getTask()->m_phase] = initTask;
          ASSERT(initTask->getRequires().size() == 0)
          initTask = nullptr;
        } else if (initTask->getRequires().size() ==
                   0) { // no ext. dependencies, then skip MPI sends
          initTask->markInitiated();
          initTask
              ->checkExternalDepCount(); // where tasks get added to external
                                         // ready queue (no ext deps though)
          initTask = nullptr;
        } else {
          havework = true;
          break;
        }
      }

      else if (usingDevice) {
        /*
         * (1.4)
         *
         * Check if highest priority GPU task's asynchronous H2D copies are
         * completed. If so, then reclaim the streams and events it used for
         * these operations, and mark as valid the vars for which this task was
         * responsible.  (If the vars are awaiting ghost cells then those vars
         * will be updated with a status to reflect they aren't quite valid yet)
         *
         * gpuVerifyDataTransferCompletion = true
         */
        if (m_detailed_tasks->getDeviceValidateRequiresCopiesTask(readyTask)) {
          gpuValidateRequiresCopies = true;
          havework = true;
          break;
        }
        /*
         * (1.4.1)
         *
         * Check if all vars and staging vars needed for ghost cell copies are
         * present and valid. If so, start the ghost cell gathering.  Otherwise,
         * put it back in this pool and try again later. gpuPerformGhostCopies =
         * true
         */
        else if (m_detailed_tasks->getDevicePerformGhostCopiesTask(readyTask)) {
          gpuPerformGhostCopies = true;
          havework = true;
          break;
        }
        /*
         * (1.4.2)
         *
         * Prevously the task was gathering in ghost vars.  See if one of those
         * tasks is done, and if so mark the vars it was processing as valid
         * with ghost cells. gpuValidateGhostCopies = true
         */
        else if (m_detailed_tasks->getDeviceValidateGhostCopiesTask(readyTask)) {
          gpuValidateGhostCopies = true;
          havework = true;
          break;
        }
        /*
         * (1.4.3)
         *
         * Check if all GPU variables for the task are either valid or valid and
         * awaiting ghost cells. If any aren't yet at that state (due to another
         * task having not copied it in yet), then repeat this step.  If all
         * variables have been copied in and some need ghost cells, then process
         * that.  If no variables need to have their ghost cells processed, the
         * GPU to GPU ghost cell copies.  Also make GPU data as being valid as
         * it is now copied into the device.
         *
         * gpuCheckIfExecutable = true
         */
        else if (m_detailed_tasks->getDeviceCheckIfExecutableTask(readyTask)) {
          gpuCheckIfExecutable = true;
          havework = true;
          break;
        }
        /*
         * (1.4.4)
         *
         * Check if highest priority GPU task's asynchronous device to device
         * ghost cell copies are finished. If so, then reclaim the streams and
         * events it used for these operations, execute the task and then put it
         * into the GPU completion-pending queue.
         *
         * gpuRunReady = true
         */
        else if (m_detailed_tasks->getDeviceReadyToExecuteTask(readyTask)) {
          gpuRunReady = true;
          havework = true;
          break;
        }
        /*
         * (1.4.5)
         *
         * Check if a CPU task needs data into host memory from GPU memory
         * If so, copies data D2H.  Also checks if all data has arrived and is
         * ready to process.
         *
         * cpuValidateRequiresCopies = true
         */
        else if (m_detailed_tasks->getHostValidateRequiresCopiesTask(readyTask)) {
          cpuValidateRequiresCopies = true;
          havework = true;
          break;
        }
        /*
         * (1.4.6)
         *
         * Check if all CPU variables for the task are either valid or valid and
         * awaiting ghost cells. If so, this task can be executed. If not,
         * (perhaps due to another task having not completed a D2H yet), then
         * repeat this step.
         *
         * cpuCheckIfExecutable = true
         */
        else if (m_detailed_tasks->getHostCheckIfExecutableTask(readyTask)) {
          cpuCheckIfExecutable = true;
          havework = true;
          break;
        }
        /*
         * (1.4.7)
         *
         * Check if highest priority GPU task's asynchronous D2H copies are
         * completed. If so, execute the task and then put it into the CPU
         * completion-pending queue.
         *
         * cpuRunReady = true
         */
        else if (m_detailed_tasks->getHostReadyToExecuteTask(readyTask)) {
          markTaskConsumed(m_num_tasks_done, m_curr_phase, m_num_phases,
                           readyTask);
          cpuRunReady = true;
          havework = true;
          break;
        }
        /*
         * (1.5)
         *
         * Check to see if any GPU tasks have been completed. This means the
         * kernel(s) have executed (which prevents out of order kernels, and
         * also keeps tasks that depend on its data to wait until the async
         * kernel call is done). This task's MPI sends can then be posted and
         * done() can be called.
         *
         * gpuPending = true
         */
        else if (m_detailed_tasks->getDeviceExecutionPendingTask(readyTask)) {
          havework = true;
          gpuPending = true;
          markTaskConsumed(m_num_tasks_done, m_curr_phase, m_num_phases,
                           readyTask);
          break;
        }
      }
      /*
       * (1.6)
       *
       * Otherwise there's nothing to do but process MPI recvs.
       */
      if (!havework) {
        if (m_recvs.size() != 0u) {
          havework = true;
          break;
        }
      }
      if (m_num_tasks_done == m_num_tasks) {
        break;
      }
    } // end while (!havework)
    // g_scheduler_mutex.unlock();

    // ----------------------------------------------------------------------------------
    // Part 2
    //    Concurrent Part:
    //      Each thread does its own thing here... modify this code with caution
    // ----------------------------------------------------------------------------------

    // ABB: this if(initTask), elseif(readyTask) is same as KokkosOMPScheduler
    if (initTask != nullptr) {
      MPIScheduler::initiateTask(initTask, m_abort, m_abort_point,
                                 m_curr_iteration);

      initTask->markInitiated();
      initTask->checkExternalDepCount();
    } else if (readyTask) {

      if (readyTask->getTask()->getType() == Task::Reduction) {
        MPIScheduler::initiateReduction(readyTask);
      }
      else if (gpuInitReady) {
        // prepare to run a GPU task.

        // Ghost cells from CPU same  device to variable not yet on GPU -> Managed already by getGridVar()
        // Ghost cells from CPU same  device to variable already on GPU -> Managed in initiateH2DCopies(), then copied with performInternalGhostCellCopies()
        // Ghost cells from GPU other device to variable not yet on GPU -> new MPI code and getGridVar()
        // Ghost cells from GPU other device to variable already on GPU -> new MPI code, then initiateH2DCopies(), and copied with performInternalGhostCellCopies()
        // Ghost cells from GPU same  device to variable not yet on GPU -> managed in initiateH2DCopies(), and copied with performInternalGhostCellCopies()
        // Ghost cells from GPU same  device to variable already on GPU -> Managed in initiateH2DCopies()?

        assignDevicesAndStreams(readyTask);
        initiateH2DCopies(readyTask);
        syncTaskGpuDWs(readyTask);

        // Determine which queue it should go into.
        // TODO: Skip queues if possible, not all tasks performed copies or
        // ghost cell gathers
        m_detailed_tasks->addDeviceValidateRequiresCopies(readyTask);

      } else if (gpuValidateRequiresCopies) {
        // Mark all requires vars this task is responsible for copying in as
        // valid.
        markDeviceRequiresDataAsValid(readyTask);
        m_detailed_tasks->addDevicePerformGhostCopies(readyTask);
      } else if (gpuPerformGhostCopies) {
        // make sure all staging vars are valid before copying ghost cells in
        if (ghostCellsProcessingReady(readyTask)) {
          performInternalGhostCellCopies(readyTask);
          m_detailed_tasks->addDeviceValidateGhostCopies(readyTask);
        } else {
          // Another task must still be copying them.  Put it back in the pool.
          m_detailed_tasks->addDevicePerformGhostCopies(readyTask);
        }
      } else if (gpuValidateGhostCopies) {
        markDeviceGhostsAsValid(readyTask);
        m_detailed_tasks->addDeviceCheckIfExecutable(readyTask);
      } else if (gpuCheckIfExecutable) {
        if (allGPUVarsProcessingReady(readyTask)) {
          // It's ready to execute.
          m_detailed_tasks->addDeviceReadyToExecute(readyTask);
        } else {
          // Not all ghost cells are ready. Another task must still be working
          // on it. Put it back in the pool.
          m_detailed_tasks->addDeviceCheckIfExecutable(readyTask);
        }
      } else if (gpuRunReady) {

        // Run the task on the GPU!
        runTask(readyTask, m_curr_iteration, thread_id, Task::GPU);

        // See if we're dealing with 32768 ghost cells per patch.  If so,
        // it's easier to manage them on the host for now than on the GPU.  We
        // can issue these on the same stream as runTask, and it won't process
        // until after the GPU kernel completed.
        // initiateD2HForHugeGhostCells(readyTask);

        m_detailed_tasks->addDeviceExecutionPending(readyTask);

      } else if (gpuPending) {
        // The GPU task has completed. All of the computes data is now valid and
        // should be marked as such.

        // Go through all computes for the task. Mark them as valid.
        markDeviceComputesDataAsValid(readyTask);

        // The Task GPU Datawarehouses are no longer needed.  Delete them on the
        // host and device.
        readyTask->deleteTaskGpuDataWarehouses();
        readyTask->deleteTemporaryTaskVars();

        // Run post GPU part of task.  It won't actually rerun the task
        // But it will run post computation management logic, which includes
        // marking the task as done.
        runTask(readyTask, m_curr_iteration, thread_id, Task::postGPU);
      }
      else {
        // prepare to run a CPU task.
        if (cpuInitReady) {

          // Some CPU tasks still interact with the GPU.  For example,
          // DataArchiver,::ouputVariables, or RMCRT task which copies over old
          // data warehouse variables to the new data warehouse, or even CPU
          // tasks which locally invoke their own quick self contained kernels
          // for quick and dirty local code which use the GPU in a way that the
          // data warehouse or the scheduler never needs to know about it (e.g.
          // transferFrom()). So because we aren't sure which CPU tasks could
          // use the GPU, just go ahead and assign each task a GPU stream.
          // assignStatusFlagsToPrepareACpuTask(readyTask);
          assignDevicesAndStreams(readyTask);

          // Run initiateD2H on all tasks in case the data we need is in GPU
          // memory but not in host memory. The exception being we don't run an
          // output task in a non-output timestep. (It would be nice if the task
          // graph didn't have this OutputVariables task if it wasn't going to
          // output data, but that would require more task graph recompilations,
          // which can be even costlier overall.  So we do the check here.)

          // ARS NOTE: Outputing and Checkpointing may be done out of
          // snyc now. I.e. turned on just before it happens rather
          // than turned on before the task graph execution.  As such,
          // one should also be checking:

          // m_application->activeReductionVariable( "outputInterval" );
          // m_application->activeReductionVariable( "checkpointInterval" );

          // However, if active the code below would be called regardless
          // if an output or checkpoint time step or not. Not sure that is
          // desired but not sure of the effect of not calling it and doing
          // an out of sync output or checkpoint.

          if ((m_output->isOutputTimeStep() ||
               m_output->isCheckpointTimeStep()) ||
              ((readyTask->getTask()->getName() !=
                "DataArchiver::outputVariables") &&
               (readyTask->getTask()->getName() !=
                "DataArchiver::outputVariables(checkpoint)"))) {
            initiateD2H(readyTask);
          }
          if (readyTask->getVarsBeingCopiedByTask().getMap().empty()) {
            if (allHostVarsProcessingReady(readyTask)) {
              m_detailed_tasks->addHostReadyToExecute(readyTask);
              // runTask(readyTask, m_curr_iteration, thread_id, Task::CPU);
              // GPUMemoryPool::reclaimGpuStreamsIntoPool(readyTask);
            } else {
              m_detailed_tasks->addHostCheckIfExecutable(readyTask);
            }
          } else {
            // for (std::multimap<GpuUtilities::LabelPatchMatlLevelDw,
            // DeviceGridVariableInfo>::iterator it =
            // readyTask->getVarsBeingCopiedByTask().getMap().begin(); it !=
            // readyTask->getVarsBeingCopiedByTask().getMap().end(); ++it) {
            // }
            // Once the D2H transfer is done, we mark those vars as valid.
            m_detailed_tasks->addHostValidateRequiresCopies(readyTask);
          }
        } else if (cpuValidateRequiresCopies) {
          markHostRequiresDataAsValid(readyTask);
          if (allHostVarsProcessingReady(readyTask)) {
            m_detailed_tasks->addHostReadyToExecute(readyTask);
            // runTask(readyTask, m_curr_iteration, thread_id, Task::CPU);
            // GPUMemoryPool::reclaimGpuStreamsIntoPool(readyTask);
          } else {
            m_detailed_tasks->addHostCheckIfExecutable(readyTask);
          }
        } else if (cpuCheckIfExecutable) {
          if (allHostVarsProcessingReady(readyTask)) {
            m_detailed_tasks->addHostReadyToExecute(readyTask);
            // runTask(readyTask, m_curr_iteration, thread_id, Task::CPU);
            // GPUMemoryPool::reclaimGpuStreamsIntoPool(readyTask);
          } else {
            // Some vars aren't valid and ready,  We must be waiting on another
            // task to finish copying in some of the variables we need.
            m_detailed_tasks->addHostCheckIfExecutable(readyTask);
          }
        } else if (cpuRunReady) {

          // run CPU task.
          runTask(readyTask, m_curr_iteration, thread_id, Task::CPU);

          // See note above near cpuInitReady.  Some CPU tasks may internally
          // interact with GPUs without modifying the structure of the data
          // warehouse. GPUMemoryPool::reclaimGpuStreamsIntoPool(readyTask);
        }
      }
    } else {
      if (m_recvs.size() != 0u) {
        MPIScheduler::processMPIRecvs(TEST);
      }
    }
  } // end while (numTasksDone < ntasks)
  ASSERT(m_num_tasks_done == m_num_tasks);
}

//______________________________________________________________________
//
void SYCLScheduler::prepareGpuDependencies(
    DetailedTask *dtask, DependencyBatch *batch, const VarLabel *pos_var,
    OnDemandDataWarehouse *dw, OnDemandDataWarehouse *old_dw,
    const DetailedDep *dep, DeviceVarDest dest) {

  // This should handle the following scenarios:
  // GPU -> different GPU same node    (write to GPU array, move to other device memory, copy in via copyGPUGhostCellsBetweenDevices)
  // GPU -> different GPU another node (write to GPU array, move to host memory, copy via MPI)
  // GPU -> CPU another node           (write to GPU array, move to host memory, copy via MPI)
  // It should not handle GPU -> CPU same node (handled in initateH2D)
  // GPU -> same GPU same node (handled in initateH2D)

  // This method therefore indicates that staging/contiguous arrays are needed,
  // and what ghost cell copies need to occur in the GPU.

  if (dep->isNonDataDependency()) {
    return;
  }

  const VarLabel *label = dep->m_req->m_var;
  const Patch *fromPatch = dep->m_from_patch;
  const int matlIndx = dep->m_matl;
  const Level *level = fromPatch->getLevel();
  const int levelID = level->getID();

  // TODO: Ask Alan about everything in the dep object.
  // the toTasks (will there be more than one?)
  // the dep->comp (computes?)
  // the dep->req (requires?)

  DetailedTask *toTask = nullptr;
  // Go through all toTasks
  for (const auto iter : dep->m_to_tasks) {
    toTask = (*iter);

    constHandle<PatchSubset> patches = toTask->getPatches();
    const int numPatches = patches->size();

    for (int i = 0; i < numPatches; i++) {
      const Patch *toPatch = patches->get(i);

      const int fromresource = dtask->getAssignedResourceIndex();
      const int toresource = toTask->getAssignedResourceIndex();

      const int fromDeviceIndex = GpuUtilities::getGpuIndexForPatch(fromPatch);
      // for now, assume that task will only work on one device
      const int toDeviceIndex = GpuUtilities::getGpuIndexForPatch(toTask->getPatches()->get(0));

      if ((fromresource == toresource) && (fromDeviceIndex == toDeviceIndex)) {
        // don't handle GPU -> same GPU same node here
        continue;
      }

      GPUDataWarehouse *gpudw = nullptr;
      if (fromDeviceIndex != -1) {
        gpudw = dw->getGPUDW(fromDeviceIndex);
        if (!gpudw->isValidOnGPU(label->getName().c_str(), fromPatch->getID(),
                                 matlIndx, levelID)) {
          continue;
        }
      } else {
        SCI_THROW(InternalError("Device index not found for " +
				label->getFullName(matlIndx, fromPatch),
                                __FILE__, __LINE__));
      }

      switch (label->typeDescription()->getType()) {
      case TypeDescription::ParticleVariable: {
      } break;
      case TypeDescription::NCVariable:
      case TypeDescription::CCVariable:
      case TypeDescription::SFCXVariable:
      case TypeDescription::SFCYVariable:
      case TypeDescription::SFCZVariable: {

        // TODO, This compiles a list of regions we need to copy into contiguous
        // arrays. We don't yet handle a scenario where the ghost cell region is
        // exactly the same size as the variable, meaning we don't need to
        // create an array and copy to it.

        // We're going to copy the ghost vars from the source variable (already
        // in the GPU) to a destination array (not yet in the GPU).  So make
        // sure there is a destination.

        // See if we're already planning on making this exact copy.  If so,
        // don't do it again.
        IntVector host_low, host_offset, host_size;
        host_low = dep->m_low;
        host_offset = dep->m_low;
        host_size = dep->m_high - dep->m_low;
        const std::size_t elementDataSize =
            OnDemandDataWarehouse::getTypeDescriptionSize(
                dep->m_req->m_var->typeDescription()->getSubType()->getType());
        const std::size_t memSize =
            host_size.x() * host_size.y() * host_size.z() * elementDataSize;
        // If this staging var already exists, then assume the full ghost cell
        // copying information has already been set up previously.  (Duplicate
        // dependencies show up by this point, so just ignore the duplicate).

        // TODO, this section should be treated atomically.  Duplicates do
        // happen, and we don't yet handle if two of the duplicates try to get
        // added to dtask->getDeviceVars().add() simultaneously.

        // NOTE: On the CPU, a ghost cell face may be sent from patch A to patch
        // B, while a ghost cell edge/line may be sent from patch A to patch C,
        // and the line of data for C is wholly within the face data for B. For
        // the sake of preparing for cuda aware MPI, we still want to create two
        // staging vars here, a contiguous face for B, and a contiguous
        // edge/line for C.
       if (!(dtask->getDeviceVars().stagingVarAlreadyExists(
                dep->m_req->m_var, fromPatch, matlIndx, levelID, host_low,
                host_size, dep->m_req->mapDataWarehouse()))) {

          // TODO: This host var really should be created last minute only if
          // it's copying data to host.  Not here.
          // TODO: Verify this cleans up.  If so change the comment.
          GridVariableBase *tempGhostVar = dynamic_cast<GridVariableBase *>(
              label->typeDescription()->createInstance());
          tempGhostVar->allocate(dep->m_low, dep->m_high);

          // Indicate we want a staging array in the device.
          dtask->getDeviceVars().add(fromPatch, matlIndx, levelID, true,
                                     host_size, memSize, elementDataSize,
                                     host_offset, dep->m_req, Ghost::None, 0,
                                     fromDeviceIndex, tempGhostVar, dest);

          // let this Task GPU DW know about this staging array
          dtask->getTaskVars().addTaskGpuDWStagingVar(
              fromPatch, matlIndx, levelID, host_offset, host_size,
              elementDataSize, dep->m_req, fromDeviceIndex);

          // Now make sure the Task DW knows about the non-staging variable
          // where the staging variable's data will come from. Scenarios occur
          // in which the same source region is listed to send to two different
          // patches. This task doesn't need to know about the same source
          // twice.
          if (!(dtask->getTaskVars().varAlreadyExists(
                  dep->m_req->m_var, fromPatch, matlIndx, levelID,
                  dep->m_req->mapDataWarehouse()))) {
            // let this Task GPU DW know about the source location.
            dtask->getTaskVars().addTaskGpuDWVar(fromPatch, matlIndx, levelID,
                                                 elementDataSize, dep->m_req,
                                                 fromDeviceIndex);
          } else {
            std::cout << myRankThread()
                      << " prepareGpuDependencies - Already had a task GPUDW "
                         "Var for label "
                      << dep->m_req->m_var->getName() << " patch "
                      << fromPatch->getID() << " matl " << matlIndx << " level "
                      << levelID << std::endl;
          }

          // Handle a GPU-another GPU same device transfer. We have already
          // queued up the staging array on source GPU. Now queue up the
          // staging array on the destination GPU.
          if (dest == GpuUtilities::anotherDeviceSameMpiRank) {
            // Indicate we want a staging array in the device.
            // TODO: We don't need a host array, it's going GPU->GPU. So get
            // rid of tempGhostVar here.
            dtask->getDeviceVars().add(
                toPatch, matlIndx, levelID, true, host_size,
                tempGhostVar->getDataSize(), elementDataSize, host_offset,
                dep->m_req, Ghost::None, 0, toDeviceIndex, tempGhostVar, dest);

            // And the task should know of this staging array.
            dtask->getTaskVars().addTaskGpuDWStagingVar(
                toPatch, matlIndx, levelID, host_offset, host_size,
                elementDataSize, dep->m_req, toDeviceIndex);
          }

          // we always write this to a "foreign" staging variable. We are going
          // to copying it from the foreign = false var to the foreign = true
          // var. Thus the patch source and destination are the same, and it's
          // staying on device.
          IntVector temp(0, 0, 0);
          dtask->getGhostVars().add(
              dep->m_req->m_var, fromPatch, fromPatch, matlIndx, levelID, false,
              true, host_offset, host_size, dep->m_low, dep->m_high,
              OnDemandDataWarehouse::getTypeDescriptionSize(
                  dep->m_req->m_var->typeDescription()
                      ->getSubType()
                      ->getType()),
              dep->m_req->m_var->typeDescription()->getSubType()->getType(),
              temp, fromDeviceIndex, toDeviceIndex, fromresource, toresource,
              (Task::WhichDW)dep->m_req->mapDataWarehouse(),
              GpuUtilities::sameDeviceSameMpiRank);

          if (dest == GpuUtilities::anotherDeviceSameMpiRank) {
            // GPU to GPU copies needs another entry indicating a peer to peer
            // transfer.

            dtask->getGhostVars().add(
                dep->m_req->m_var, fromPatch, toPatch, matlIndx, levelID, true,
                true, host_offset, host_size, dep->m_low, dep->m_high,
                OnDemandDataWarehouse::getTypeDescriptionSize(
                    dep->m_req->m_var->typeDescription()
                        ->getSubType()
                        ->getType()),
                dep->m_req->m_var->typeDescription()->getSubType()->getType(),
                temp, fromDeviceIndex, toDeviceIndex, fromresource, toresource,
                (Task::WhichDW)dep->m_req->mapDataWarehouse(),
                GpuUtilities::anotherDeviceSameMpiRank);

          } else if (dest == GpuUtilities::anotherMpiRank) {
            dtask->getGhostVars().add(
                dep->m_req->m_var, fromPatch, toPatch, matlIndx, levelID, true,
                true, host_offset, host_size, dep->m_low, dep->m_high,
                OnDemandDataWarehouse::getTypeDescriptionSize(
                    dep->m_req->m_var->typeDescription()
                        ->getSubType()
                        ->getType()),
                dep->m_req->m_var->typeDescription()->getSubType()->getType(),
                temp, fromDeviceIndex, toDeviceIndex, fromresource, toresource,
                (Task::WhichDW)dep->m_req->mapDataWarehouse(),
                GpuUtilities::anotherMpiRank);
          }
        }
      } break;
      default: {
        std::cerr << "SYCLScheduler::prepareGPUDependencies(), unsupported "
                     "variable type"
                  << std::endl;
      }
      }
    } // numPatches
  } // to_tasks
}

//______________________________________________________________________
//
void SYCLScheduler::gpuInitialize(bool reset) {
  int numDevices = 0;
  syclGetDeviceCount(&numDevices);
  m_num_devices = numDevices;
  syclSetDevice(0);
}

//______________________________________________________________________
//
void SYCLScheduler::turnIntoASuperPatch(
    GPUDataWarehouse *const gpudw, const Level *const level,
    const IntVector &low, const IntVector &high, const VarLabel *const label,
    const Patch *const patch, const int matlIndx, const int levelID) {
  // Handle superpatch stuff
  // This was originally designed for the use case of turning an entire level
  // into a variable. We need to set up the equivalent of a super patch. For
  // example, suppose a simulation has 8 patches and 2 ranks and 1 level, and
  // this rank owns patches 0, 1, 2, and 3.  Further suppose this scheduler
  // thread is checking to see the status of a patch 1 variable which has a ton
  // of ghost cells associated with it, enough to envelop all seven other
  // patches.  Also suppose patch 1 is found on the CPU, ghost cells for patches
  // 4, 5, 6, and 7 have previously been sent to us, patch 1 is needed on the
  // GPU, and this is the first thread to process this situation. This thread's
  // job should be to claim it is responsible for processing the variable for
  // patches 0, 1, 2, and 3.  Four GPU data warehouse entries should be created,
  // one for each patch.

  // Patches 0, 1, 2, and 3 should be given the same pointer, same low, same
  // high, (TODO: but different offsets). In order to avoid concurrency problems
  // when marking all patches in the superpatch region as belonging to the
  // superpatch, we need to avoid Dining Philosophers problem.  That is
  // accomplished by claiming patches in *sorted* order, and no scheduler thread
  // can attempt to claim any later patch if it hasn't yet claimed a former
  // patch.  The first thread to claim all will have claimed the "superpatch"
  // region.

  // Superpatches essentially are just windows into a shared variable, it uses
  // shared_ptrs behind the scenes With this later only one alloaction or H2D
  // transfer can be done.  This method's job is just to concurrently set up all
  // the underlying shared_ptr work.

  // Note: Superpatch approaches won't work if for some reason a prior task
  // copied a patch in a non-superpatch manner, at the current moment no known
  // simulation will ever do this.  It is also why we try to prepare the
  // superpatch a bit upstream before concurrency checks start, and not down in
  // prepareDeviceVars(). Brad P - 8/6/2016 Future note:   A lock free reference
  // counter should also be created and set to 4 for the above example.
  //  If a patch is "vacated" from the GPU, the reference counter should be
  //  reduced.  If it hits 0, it
  // shouldn't be automatically deleted, but only available for removal if the
  // memory space hits capacity.

  bool thisThreadHandlesSuperPatchWork = false;
  char label_cstr[80];
  strcpy(label_cstr, label->getName().c_str());

  // Get all patches in the superpatch. Assuming our superpatch is the entire
  // level. This also sorts the neighbor patches by ID for us.  Note that if the
  // current patch is smaller than all the neighbors, we have to work that in
  // too.

  Patch::selectType neighbors;
  // IntVector low, high;
  // level->computeVariableExtents(type, low, high);  //Get the low and high for
  // the level
  level->selectPatches(low, high, neighbors);

  // mark the lowest patch as being the superpatch
  const Patch *firstPatchInSuperPatch = nullptr;
  if (neighbors.size() == 0) {
    // this must be a one patch simulation, there are no neighbors.
    firstPatchInSuperPatch = patch;
  } else {
    firstPatchInSuperPatch = neighbors[0]->getRealPatch();
    // seeing if this patch is lower in ID number than the neighbor patches.
    if (patch->getID() < firstPatchInSuperPatch->getID()) {
      firstPatchInSuperPatch = patch;
    }
  }

  // The firstPatchInSuperPatch may not have yet been handled by a prior task
  // (such as it being a patch assigned to a different node).  So make an entry
  // if needed.
  gpudw->putUnallocatedIfNotExists(label_cstr, firstPatchInSuperPatch->getID(),
                                   matlIndx, levelID, false, make_int3(0,0,0), make_int3(0,0,0));
  thisThreadHandlesSuperPatchWork = gpudw->compareAndSwapFormASuperPatchGPU(
      label_cstr, firstPatchInSuperPatch->getID(), matlIndx, levelID);

  // At this point the patch has been marked as a superpatch.

  if (thisThreadHandlesSuperPatchWork) {

    gpudw->setSuperPatchLowAndSize(
        label_cstr, firstPatchInSuperPatch->getID(), matlIndx, levelID,
        make_int3(low.x(), low.y(), low.z()),
        make_int3(high.x() - low.x(), high.y() - low.y(), high.z() - low.z()));

    // This thread turned the lowest ID'd patch in the region into a superpatch.
    // Go through *neighbor* patches in the superpatch region and flag them as
    // being a superpatch as well (the copySuperPatchInfo call below can also
    // flag it as a superpatch.
    for (unsigned int i = 0; i < neighbors.size(); i++) {
      if (neighbors[i]->getRealPatch() !=
          firstPatchInSuperPatch) { // This if statement is because there is no
                                    // need to merge itself

        // These neighbor patches may not have yet been handled by a prior task.
        // So go ahead and make sure they show up in the database
        gpudw->putUnallocatedIfNotExists(
            label_cstr, neighbors[i]->getRealPatch()->getID(), matlIndx,
            levelID, false, make_int3(0, 0, 0), make_int3(0, 0, 0));

        // TODO: Ensure these variables weren't yet allocated, in use, being
        // copied in, etc. At the time of writing, this scenario didn't exist.
        // Some ways to solve this include 1) An "I'm using this" reference
        // counter. 2) Moving superpatch creation to the start of a timestep,
        // and not at the start of initiateH2D, or 3) predetermining at the
        // start of a timestep what superpatch regions will be, and then we can
        // just form them together here

        // Shallow copy this neighbor patch into the superaptch
        gpudw->copySuperPatchInfo(label_cstr, firstPatchInSuperPatch->getID(),
                                  neighbors[i]->getRealPatch()->getID(),
                                  matlIndx, levelID);
      }
    }
    gpudw->compareAndSwapSetSuperPatchGPU(
        label_cstr, firstPatchInSuperPatch->getID(), matlIndx, levelID);

  } else {
    // spin and wait until it's done.
    while (!gpudw->isSuperPatchGPU(label_cstr, firstPatchInSuperPatch->getID(),
                                   matlIndx, levelID))
      ;
  }
}

//______________________________________________________________________
//
// initiateH2DCopies is a key method for the GPU Data Warehouse and the SYCL
// Scheduler It helps manage which data needs to go H2D, what allocations and
// ghost cells need to be copied, etc. It also manages concurrency so that no
// two threads could process the same task. A general philosophy is that this
// section should do atomic compareAndSwaps if it find it is the one to
// allocate, copy in, or copy in with ghosts.  After any of those actions are
// seen to have completed then they can get marked as being "allocated", "copied-in"
// , or "copied-in with ghosts".

void SYCLScheduler::initiateH2DCopies(DetailedTask *dtask) {

  const Task *task = dtask->getTask();
  dtask->clearPreparationCollections();

  // Gather up all possible dependents from requires and computes and remove
  // duplicates (we don't want to transfer some variables twice). Note: A task
  // can only run on one level at a time.  It could run multiple patches and
  // multiple materials, but a single task will never run multiple levels.
  std::map<labelPatchMatlDependency, const Task::Dependency *> vars;
  for (const Task::Dependency *dependantVar = task->getRequires(); dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(dtask->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(),
                                      patches->get(i)->getID(), matls->get(j),
                                      Task::Requires);
        if (vars.find(lpmd) == vars.end()) {
          vars.insert(std::map<labelPatchMatlDependency, const Task::Dependency *>::value_type(lpmd, dependantVar));
        }
      }
    }
  }
  for (const Task::Dependency *dependantVar = task->getComputes(); dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches = dependantVar->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls = dependantVar->getMaterialsUnderDomain(dtask->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(),
                                      patches->get(i)->getID(), matls->get(j),
                                      Task::Computes);
        if (vars.find(lpmd) == vars.end()) {
          vars.insert(std::map<labelPatchMatlDependency, const Task::Dependency *>::value_type(lpmd, dependantVar));
        }
      }
    }
  }

  // The task runs on one device. SYCLScheduler::assignDevicesAndStreams()
  // set this info so query accordingly.
  auto deviceIDSet = dtask->getDeviceNums();

  // Go through each unique dependent var and see if we should allocate space
  // and/or queue it to be copied H2D.
  for (auto varIter : vars) {

    const Task::Dependency *curDependency = varIter->second;
    const TypeDescription::Type type = curDependency->m_var->typeDescription()->getType();

    // make sure we're dealing with a variable we support
    if (type == TypeDescription::CCVariable ||
        type == TypeDescription::NCVariable ||
        type == TypeDescription::SFCXVariable ||
        type == TypeDescription::SFCYVariable ||
        type == TypeDescription::SFCZVariable ||
        type == TypeDescription::PerPatch ||
        type == TypeDescription::ReductionVariable) {

      constHandle<PatchSubset> patches = curDependency->getPatchesUnderDomain(dtask->getPatches());
      const int numPatches = patches->size();
      const int patchID = varIter->first.m_patchID;
      const Patch *patch = nullptr;
      const Level *level = nullptr;
      for (int i = 0; i < numPatches; i++) {
        if (patches->get(i)->getID() == patchID) {
          patch = patches->get(i);
          level = patch->getLevel();
        }
      }
      if (!patch) {
        SCI_THROW( InternalError("SYCLScheduler::initiateD2H() patch not found.", __FILE__, __LINE__));
      }
      const int matlID = varIter->first.m_matlIndex;
      int levelID = level->getID();
      if (curDependency->m_var->typeDescription()->getType() == TypeDescription::ReductionVariable) {
        levelID = -1;
      }

      const int deviceIndex = GpuUtilities::getGpuIndexForPatch(patch);

      // For this dependency, get its CPU Data Warehouse and GPU Datawarehouse.
      const int dwIndex = curDependency->mapDataWarehouse();
      OnDemandDataWarehouseP dw = m_dws[dwIndex];
      GPUDataWarehouse *gpudw = dw->getGPUDW(deviceIndex);

      // a fix for when INF ghost cells are requested such as in RMCRT e.g.
      // tsk->requires(abskg_dw, d_abskgLabel, gac, SHRT_MAX);
      bool uses_SHRT_MAX = (curDependency->m_num_ghost_cells == SHRT_MAX);

      // Get all size information about this dependency.
      IntVector low, high;
      if (uses_SHRT_MAX) {
        level->computeVariableExtents(type, low, high);
      } else {
        Patch::VariableBasis basis = Patch::translateTypeToBasis(type, false);
        patch->computeVariableExtents(
            basis, curDependency->m_var->getBoundaryLayer(),
            curDependency->m_gtype, curDependency->m_num_ghost_cells, low,
            high);
      }
      const IntVector host_size = high - low;
      const std::size_t elementDataSize =
          OnDemandDataWarehouse::getTypeDescriptionSize(
              curDependency->m_var->typeDescription()->getSubType()->getType());
      std::size_t memSize = 0;
      if (type == TypeDescription::PerPatch || type == TypeDescription::ReductionVariable) {
        memSize = elementDataSize;
      } else {
        memSize = host_size.x() * host_size.y() * host_size.z() * elementDataSize;
      }

      // Set up/get status flags
      // Start by checking if an entry doesn't exist in the GPU data warehouse.
      // If so, create one.
      gpudw->putUnallocatedIfNotExists(
          curDependency->m_var->getName().c_str(), patchID, matlID, levelID,
          false, make_int3(low.x(), low.y(), low.z()),
          make_int3(host_size.x(), host_size.y(), host_size.z()));

      bool correctSize = false;
      bool allocating = false;
      bool allocated = false;
      bool copyingIn = false;
      bool validOnGPU = false;
      bool gatheringGhostCells = false;
      bool validWithGhostCellsOnGPU = false;
      bool deallocating = false;
      bool formingSuperPatch = false;
      bool superPatch = false;

      gpudw->getStatusFlagsForVariableOnGPU(
          correctSize, allocating, allocated, copyingIn, validOnGPU,
          gatheringGhostCells, validWithGhostCellsOnGPU, deallocating,
          formingSuperPatch, superPatch,
          curDependency->m_var->getName().c_str(), patchID, matlID, levelID,
          make_int3(low.x(), low.y(), low.z()),
          make_int3(host_size.x(), host_size.y(), host_size.z()));

      if (curDependency->m_dep_type == Task::Requires) {

        // For any variable, only ONE task should manage all ghost cells for it.
        // It is a giant mess to try and have two tasks simultaneously managing
        // ghost cells for a single var. So if ghost cells are required, attempt
        // to claim that we're the ones going to manage ghost cells This changes
        // a var's status to valid awaiting ghost data if this task claims
        // ownership of managing ghost cells Otherwise the var's status is left
        // alone (perhaps the ghost cells were already processed by another task
        // a while ago)
        bool gatherGhostCells = false;
        if (curDependency->m_gtype != Ghost::None && curDependency->m_num_ghost_cells > 0) {

          if (uses_SHRT_MAX) {
            // Turn this into a superpatch if not already done so:
            turnIntoASuperPatch(gpudw, level, low, high, curDependency->m_var,
                                patch, matlID, levelID);

            // At the moment superpatches are gathered together through an
            // upcoming getRegionModifiable() call.  So we still need to mark it
            // as AWAITING_GHOST_CELLS. It should trigger as one of the simpler
            // scenarios below where it knows it can gather the ghost cells
            // host-side before sending it into GPU memory.
          }

          // See if we get to be the lucky thread that processes all ghost cells
          // for this simulation variable
          gatherGhostCells = gpudw->compareAndSwapAwaitingGhostDataOnGPU(
              curDependency->m_var->getName().c_str(), patchID, matlID,
              levelID);
        }

        if ((allocating || allocated) && correctSize && (copyingIn || validOnGPU)) {
          // This variable exists or soon will exist on the destination.  So the
          // non-ghost cell part of this variable doesn't need any more work.

          // Queue it to be added to this tasks's TaskDW.
          // It's possible this variable data already was queued to be sent in
          // due to this patch being a ghost cell region of another patch So
          // just double check to prevent duplicates.
          if (!dtask->getTaskVars().varAlreadyExists(
                  curDependency->m_var, patch, matlID, levelID,
                  curDependency->mapDataWarehouse())) {
            dtask->getTaskVars().addTaskGpuDWVar(patch, matlID, levelID,
                                                 elementDataSize, curDependency,
                                                 deviceIndex);
          }

          if (gatherGhostCells) {
            // The variable's space exists or will soon exist on the GPU.  Now
            // copy in any ghost cells into the GPU and let the GPU handle the
            // ghost cell copying logic.

            // Indicate to the scheduler later on that this variable can be
            // marked as valid with ghost cells.
            dtask->getVarsToBeGhostReady().addVarToBeGhostReady(
                dtask->getName(), patch, matlID, levelID, curDependency,
                deviceIndex);

            std::vector<OnDemandDataWarehouse::ValidNeighbors> validNeighbors;
            dw->getValidNeighbors(
                curDependency->m_var, matlID, patch, curDependency->m_gtype,
                curDependency->m_num_ghost_cells, validNeighbors);
            for (auto iter : validNeighbors) {

              const Patch *sourcePatch = nullptr;
              if (iter->neighborPatch->getID() >= 0) {
                sourcePatch = iter->neighborPatch;
              } else {
                // This occurs on virtual patches.  They can be "wrap around"
                // patches, meaning if you go to one end of a domain you will
                // show up on the other side.  Virtual patches have negative
                // patch IDs, but they know what real patch they are referring
                // to.
                sourcePatch = iter->neighborPatch->getRealPatch();
              }

              IntVector ghost_host_low(0, 0, 0), ghost_host_high(0, 0, 0),
                  ghost_host_size(0, 0, 0);
              IntVector ghost_host_offset(0, 0, 0), ghost_host_strides(0, 0, 0);

              IntVector virtualOffset = iter->neighborPatch->getVirtualOffset();

              int sourceDeviceNum = GpuUtilities::getGpuIndexForPatch(sourcePatch);
              int destDeviceNum = deviceIndex;

              // Find out who has our ghost cells.  Listed in priority...
              // It could be in the GPU as a staging/foreign var
              // Or in the GPU as a full variable
              // Or in the CPU as a foreign var
              // Or in the CPU as a regular variable
              bool useGpuStaging = false;
              bool useGpuGhostCells = false;
              bool useCpuForeign = false;
              bool useCpuGhostCells = false;

              // See if it's in the GPU as a staging/foreign var
              useGpuStaging = gpudw->stagingVarExists(
                  curDependency->m_var->getName().c_str(), patchID, matlID,
                  levelID,
                  make_int3(iter->low.x(), iter->low.y(), iter->low.z()),
                  make_int3(iter->high.x() - iter->low.x(),
                            iter->high.y() - iter->low.y(),
                            iter->high.z() - iter->low.z()));

              // See if we have the entire neighbor patch in the GPU (not just a
              // staging)
              useGpuGhostCells =
                  gpudw->isValidOnGPU(curDependency->m_var->getName().c_str(),
                                      sourcePatch->getID(), matlID, levelID);

              // See if we have CPU foreign var data or just the plain CPU
              // variable we can use Note: We don't have a full system in place
              // to set valid all CPU variables.  Specifically foreign variables
              // are not set, and so the line below is commented out.  In the
              // meantime assume that if it's not on the GPU, it must be on the
              // CPU. if
              // (gpudw->isValidOnCPU(curDependency->m_var->getName().c_str(),
              // sourcePatch->getID(), matlID, levelID)) {
              if (iter->validNeighbor && iter->validNeighbor->isForeign()) {
                useCpuForeign = true;
              } else {
                useCpuGhostCells = true;
              }
              //}

              // get the sizes of the source variable
              if (useGpuStaging) {
                ghost_host_low = iter->low;
                ghost_host_high = iter->high;
                ghost_host_size = ghost_host_high - ghost_host_low;
              } else if (useGpuGhostCells) {
                GPUDataWarehouse::GhostType throwaway1;
                int throwaway2;
                sycl::int3 ghost_host_low3, ghost_host_high3, ghost_host_size3;

                gpudw->getSizes(ghost_host_low3, ghost_host_high3,
                                ghost_host_size3, throwaway1, throwaway2,
                                curDependency->m_var->getName().c_str(),
                                patchID, matlID, levelID);

                ghost_host_low =
                    IntVector(ghost_host_low3.x(), ghost_host_low3.y(),
                              ghost_host_low3.z());
                ghost_host_high =
                    IntVector(ghost_host_high3.x(), ghost_host_high3.y(),
                              ghost_host_high3.z());
                ghost_host_size =
                    IntVector(ghost_host_size3.x(), ghost_host_size3.y(),
                              ghost_host_size3.z());

              } else if (useCpuForeign || useCpuGhostCells) {
                iter->validNeighbor->getSizes(
                    ghost_host_low, ghost_host_high, ghost_host_offset,
                    ghost_host_size, ghost_host_strides);
              }
              const std::size_t ghost_mem_size =
                  ghost_host_size.x() * ghost_host_size.y() *
                  ghost_host_size.z() * elementDataSize;

              if (useGpuStaging) {

                // Make sure this task GPU DW knows about the staging var
                dtask->getTaskVars().addTaskGpuDWStagingVar(
                    patch, matlID, levelID, iter->low, iter->high - iter->low,
                    elementDataSize, curDependency, destDeviceNum);

                // Assume for now that the ghost cell region is also the exact
                // same size as the staging var.  (If in the future ghost cell
                // data is managed a bit better as it currently does on the CPU,
                // then some ghost cell regions will be found *within* an
                // existing staging var.  This is known to happen with Wasatch
                // computations involving periodic boundary scenarios.)
                dtask->getGhostVars().add(
                    curDependency->m_var, patch,
                    patch, /*We're merging the staging variable on in*/
                    matlID, levelID, true, false,
                    iter->low, /*Assuming ghost cell region is the variable size*/
                    IntVector(iter->high.x() - iter->low.x(),
                              iter->high.y() - iter->low.y(),
                              iter->high.z() - iter->low.z()),
                    iter->low, iter->high, elementDataSize,
                    curDependency->m_var->typeDescription()
                        ->getSubType()
                        ->getType(),
                    virtualOffset, destDeviceNum, destDeviceNum, -1,
                    -1, /* we're copying within a device, so destDeviceNum ->
                           destDeviceNum */
                    (Task::WhichDW)curDependency->mapDataWarehouse(),
                    GpuUtilities::sameDeviceSameMpiRank);

              } else if (useGpuGhostCells) {
                // If this task doesn't own this source patch, then we need to
                // make sure the upcoming task data warehouse at least has
                // knowledge of this GPU variable that already exists in the
                // GPU.  So queue up to load the neighbor patch metadata into
                // the task datawarehouse.
                if (!patches->contains(sourcePatch)) {
                  if (!(dtask->getTaskVars().varAlreadyExists(
                          curDependency->m_var, sourcePatch, matlID, levelID,
                          (Task::WhichDW)curDependency->mapDataWarehouse()))) {
                    dtask->getTaskVars().addTaskGpuDWVar(
                        sourcePatch, matlID, levelID, elementDataSize,
                        curDependency, sourceDeviceNum);
                  }
                }

                // Store the source and destination patch, and the range of the
                // ghost cells A GPU kernel will use this collection to do all
                // internal GPU ghost cell copies for that one specific GPU.
                dtask->getGhostVars().add(
                    curDependency->m_var, sourcePatch, patch, matlID, levelID,
                    false, false, ghost_host_low, ghost_host_size, iter->low,
                    iter->high, elementDataSize,
                    curDependency->m_var->typeDescription()
                        ->getSubType()
                        ->getType(),
                    virtualOffset, destDeviceNum, destDeviceNum, -1,
                    -1, /* we're copying within a device, so destDeviceNum ->
                           destDeviceNum */
                    (Task::WhichDW)curDependency->mapDataWarehouse(),
                    GpuUtilities::sameDeviceSameMpiRank);
              } else if (useCpuForeign) {

                // Prepare to tell the host-side GPU DW to allocate space for
                // this variable. Since we already got the gridVariableBase
                // pointer to that foreign var, go ahead and add it in here.
                // (The OnDemandDataWarehouse is weird, it doesn't let you query
                // foreign vars, it will try to inflate a regular var and deep
                // copy the foreign var on in.  So for now, just pass in the
                // pointer.)
                dtask->getDeviceVars().add(
                    sourcePatch, matlID, levelID, true, ghost_host_size,
                    ghost_mem_size, elementDataSize, ghost_host_low,
                    curDependency, Ghost::None, 0, destDeviceNum,
                    iter->validNeighbor, GpuUtilities::sameDeviceSameMpiRank);

                // Let this Task GPU DW know about this staging array.  We may
                // end up not needed it if another thread processes it or it
                // became part of a superpatch.  We'll figure that out later
                // when we go actually add it.
                dtask->getTaskVars().addTaskGpuDWStagingVar(
                    sourcePatch, matlID, levelID, ghost_host_low,
                    ghost_host_size, elementDataSize, curDependency,
                    sourceDeviceNum);

                dtask->getGhostVars().add(
                    curDependency->m_var, sourcePatch, patch, matlID, levelID,
                    true, false, ghost_host_low, ghost_host_size, iter->low,
                    iter->high, elementDataSize,
                    curDependency->m_var->typeDescription()
                        ->getSubType()
                        ->getType(),
                    virtualOffset, destDeviceNum, destDeviceNum, -1,
                    -1, /* we're copying within a device, so destDeviceNum ->
                           destDeviceNum */
                    (Task::WhichDW)curDependency->mapDataWarehouse(),
                    GpuUtilities::sameDeviceSameMpiRank);

              } else if (useCpuGhostCells) {
                // This handles the scenario where the variable is in the GPU,
                // but the ghost cell data is only found in the neighboring
                // normal patch (non-foreign) in host memory.  Ghost cells
                // haven't been gathered in or started to be gathered in.

                // Check if we should copy this patch into the GPU.

                // TODO: Instead of copying the entire patch for a ghost cell,
                // we should just create a foreign var, copy a contiguous array
                // of ghost cell data into that foreign var, then copy in that
                // foreign var.  If it's a foreign var, then the foreign var
                // section above should handle it, not here.
                if (!dtask->getDeviceVars().varAlreadyExists(
                        curDependency->m_var, sourcePatch, matlID, levelID,
                        curDependency->mapDataWarehouse())) {
                  // Prepare to tell the host-side GPU DW to possibly allocate
                  // and/or copy this variable.
                  dtask->getDeviceVars().add(
                      sourcePatch, matlID, levelID, false, ghost_host_size,
                      ghost_mem_size, elementDataSize, ghost_host_low,
                      curDependency, Ghost::None, 0, destDeviceNum, nullptr,
                      GpuUtilities::sameDeviceSameMpiRank);

                  // Prepare this task GPU DW for knowing about this variable on
                  // the GPU.
                  dtask->getTaskVars().addTaskGpuDWVar(
                      sourcePatch, matlID, levelID, elementDataSize,
                      curDependency, destDeviceNum);

                } else { // else the variable is already in deviceVars
                }

                // Add in info to perform a GPU ghost cell copy.  (It will
                // ensure duplicates can't be entered.)
                dtask->getGhostVars().add(
                    curDependency->m_var, sourcePatch, patch, matlID, levelID,
                    false, false, ghost_host_low, ghost_host_size, iter->low,
                    iter->high, elementDataSize,
                    curDependency->m_var->typeDescription()
                        ->getSubType()
                        ->getType(),
                    virtualOffset, destDeviceNum, destDeviceNum, -1,
                    -1, /* we're copying within a device, so destDeviceNum ->
                           destDeviceNum */
                    (Task::WhichDW)curDependency->mapDataWarehouse(),
                    GpuUtilities::sameDeviceSameMpiRank);
              } else {
                SCI_THROW(InternalError(
                    "Needed ghost cell data not found on the CPU or a GPU\n",
                    __FILE__, __LINE__));
              }
            } // end neighbors for loop
          }   // end if(gatherGhostCells)
        } else if ((allocated || allocating) && !correctSize) {
          // At the moment this isn't allowed. So it does an exit(-1).  There
          // are two reasons for this. First, the current CPU system always
          // needs to "resize" variables when ghost cells are required.
          // Essentially the variables weren't created with room for ghost
          // cells, and so room  needs to be created. This step can be somewhat
          // costly (I've seen a benchmark where it took 5% of the total
          // computation time). And at the moment this hasn't been coded to
          // resize on the GPU.  It would require an additional step and
          // synchronization to make it work.
          // The second reason is with concurrency.  Suppose a patch that CPU
          // thread A own needs ghost cells from a patch that CPU thread B owns.
          // A can recognize that B's data is valid on the GPU, and so it stores
          // for the future to copy B's data on in.  Meanwhile B notices it
          // needs to resize.  So A could start trying to copy in B's ghost cell
          // data while B is resizing its own data. I believe both issues can be
          // fixed with proper checkpoints.  But in reality we shouldn't be
          // resizing variables on the GPU, so this event should never happen.
          gpudw->remove(curDependency->m_var->getName().c_str(), patchID,
                        matlID, levelID);
          std::cerr
              << "Resizing of GPU grid vars not implemented at this time.  "
              << "For the GPU, computes need to be declared with scratch "
                 "computes to have room for ghost cells."
              << "Requested var of size (" << host_size.x() << ", "
              << host_size.y() << ", " << host_size.z() << ") "
              << "with offset (" << low.x() << ", " << low.y() << ", "
              << low.z() << ")" << std::endl;
          exit(-1);

        } else if ((!allocated && !allocating) ||
                   ((allocated || allocating) && correctSize && !validOnGPU &&
                    !copyingIn)) {

          // It's either not on the GPU, or space exists on the GPU for it but
          // it is invalid. Either way, gather all ghost cells host side (if
          // needed), then queue the data to be copied in H2D.  If the data
          // doesn't exist in the GPU, then the upcoming allocateAndPut will
          // allocate space for it.  Otherwise if it does exist on the GPU, the
          // upcoming allocateAndPut will notice that and simply configure it to
          // reuse the pointer.

          if (type == TypeDescription::CCVariable ||
              type == TypeDescription::NCVariable ||
              type == TypeDescription::SFCXVariable ||
              type == TypeDescription::SFCYVariable ||
              type == TypeDescription::SFCZVariable) {

            // Queue this CPU var to go into the host-side GPU DW.
            // Also queue that this GPU DW var should also be found in this
            // tasks's Task DW.

            dtask->getDeviceVars().add(
                patch, matlID, levelID, false, host_size, memSize,
                elementDataSize, low, curDependency, curDependency->m_gtype,
                curDependency->m_num_ghost_cells, deviceIndex, nullptr,
                GpuUtilities::sameDeviceSameMpiRank);
            dtask->getTaskVars().addTaskGpuDWVar(patch, matlID, levelID,
                                                 elementDataSize, curDependency,
                                                 deviceIndex);

            // Mark that when this variable is copied in, it will have its ghost
            // cells ready too.
            if (gatherGhostCells) {
              dtask->getVarsToBeGhostReady().addVarToBeGhostReady(
                  dtask->getName(), patch, matlID, levelID, curDependency,
                  deviceIndex);
            }
          } else if (type == TypeDescription::PerPatch ||
                     type == TypeDescription::ReductionVariable) {
            if (type == TypeDescription::ReductionVariable) levelID = -1;

            dtask->getDeviceVars().add(patch, matlID, levelID, elementDataSize,
                                       elementDataSize, curDependency,
                                       deviceIndex, nullptr,
                                       GpuUtilities::sameDeviceSameMpiRank);
            dtask->getTaskVars().addTaskGpuDWVar(patch, matlID, levelID,
                                                 elementDataSize, curDependency,
                                                 deviceIndex);
          } else {
            std::cerr << "SYCLScheduler::initiateH2D(), unsupported "
                         "variable type for computes variable "
                      << curDependency->m_var->getName() << std::endl;
          }
        }
      } else if (curDependency->m_dep_type == Task::Computes) {
        // compute the amount of space the host needs to reserve on the GPU for
        // this variable.

        if (type == TypeDescription::PerPatch ||
            type == TypeDescription::ReductionVariable) {
          // For PerPatch, it's not a mesh of variables, it's just a single
          // variable, so elementDataSize is the memSize.

          // For ReductionVariable, it's not a mesh of variables, it's just a
          // single variable, so elementDataSize is the memSize.

          dtask->getDeviceVars().add(
            patch, matlID, levelID, memSize, elementDataSize, curDependency,
            deviceIndex, nullptr, GpuUtilities::sameDeviceSameMpiRank);
        } else if (type == TypeDescription::CCVariable ||
                   type == TypeDescription::NCVariable ||
                   type == TypeDescription::SFCXVariable ||
                   type == TypeDescription::SFCYVariable ||
                   type == TypeDescription::SFCZVariable) {

          dtask->getDeviceVars().add(
            patch, matlID, levelID, false, host_size, memSize,
            elementDataSize, low, curDependency, curDependency->m_gtype,
            curDependency->m_num_ghost_cells, deviceIndex, nullptr,
            GpuUtilities::sameDeviceSameMpiRank);
        } else {
          std::cerr << "SYCLScheduler::initiateH2D(), unsupported variable "
            "type for computes variable "
                    << curDependency->m_var->getName() << std::endl;
        }

        dtask->getTaskVars().addTaskGpuDWVar(patch, matlID, levelID,
                                             elementDataSize, curDependency,
                                             deviceIndex);

      }
    }
  }

  // We've now gathered up all possible things that need to go on the device.
  // Copy it over.
  createTaskGpuDWs(dtask);
  prepareDeviceVars(dtask);

  // At this point all needed variables will have a pointer.
  prepareTaskVarsIntoTaskDW(dtask);
  prepareGhostCellsIntoTaskDW(dtask);
}

//______________________________________________________________________
//
void SYCLScheduler::prepareDeviceVars(DetailedTask *dtask) {
  bool isStaging = false;

  // std::set<unsigned int> deviceNums = dtask->getDeviceNums();
  // for (std::set<unsigned int>::const_iterator deviceNums_it =
  // deviceNums.begin(); deviceNums_it != deviceNums.end(); ++deviceNums_it) {
  isStaging = false;
  // Because maps are unordered, it is possible a staging var could be inserted
  // before the regular var exists. So just loop twice, once when all staging is
  // false, then loop again when all staging is true
  for (int i = 0; i < 2; i++) {
    // Get all data in the GPU, and store it on the GPU Data Warehouse on the
    // host, as only it is responsible for management of data.  So this
    // processes the previously collected deviceVars.
    std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>
        &varMap = dtask->getDeviceVars().getMap();

    for (auto it : varMap) {
      int whichGPU = it->second.m_whichGPU;
      int dwIndex = it->second.m_dep->mapDataWarehouse();

      OnDemandDataWarehouseP dw = m_dws[dwIndex];
      GPUDataWarehouse *gpudw = dw->getGPUDW(whichGPU);
      if (!gpudw) {
        SCI_THROW(
            InternalError("No GPU data warehouse found\n", __FILE__, __LINE__));
      }

      if (it->second.m_staging == isStaging) {

        if (dtask->getDeviceVars().getTotalVars(whichGPU, dwIndex) >= 0) {

          void *device_ptr = nullptr; // device base pointer to raw data

          const IntVector offset = it->second.m_offset;
          const IntVector size = it->second.m_sizeVector;
          const IntVector low = offset;
          const IntVector high = offset + size;
          const TypeDescription *type_description =
              it->second.m_dep->m_var->typeDescription();
          const TypeDescription::Type type = type_description->getType();
          const TypeDescription::Type subtype =
              type_description->getSubType()->getType();
          const VarLabel *label = it->second.m_dep->m_var;
          char label_cstr[80];
          strcpy(label_cstr, it->second.m_dep->m_var->getName().c_str());
          const Patch *patch = it->second.m_patchPointer;
          const int patchID = it->first.m_patchID;
          const int matlIndx = it->first.m_matlIndx;
          const int levelID = it->first.m_levelIndx;
          const std::size_t elementDataSize = it->second.m_sizeOfDataType;
          const bool staging = it->second.m_staging;
          const int numGhostCells = it->second.m_numGhostCells;
          Ghost::GhostType ghosttype = it->second.m_gtype;
          bool uses_SHRT_MAX = (numGhostCells == SHRT_MAX);

          // Allocate the vars if needed.  If they've already been allocated,
          // then this simply sets the var to reuse the existing pointer.
          switch (type) {
          case TypeDescription::PerPatch:
          case TypeDescription::ReductionVariable: {
            gpudw->allocateAndPut(subtype, device_ptr, label_cstr, patchID,
                                  matlIndx, levelID, elementDataSize);
            break;
          }
          case TypeDescription::CCVariable:
          case TypeDescription::NCVariable:
          case TypeDescription::SFCXVariable:
          case TypeDescription::SFCYVariable:
          case TypeDescription::SFCZVariable: {
            sycl::int3 device_size{0, 0, 0};
            sycl::int3 device_offset{0, 0, 0};

            gpudw->allocateAndPut(
                subtype, device_ptr, device_size, device_offset, label_cstr,
                patchID, matlIndx, levelID, staging,
                make_int3(low.x(), low.y(), low.z()),
                make_int3(high.x(), high.y(), high.z()), elementDataSize,
                (GPUDataWarehouse::GhostType)(it->second.m_gtype),
                it->second.m_numGhostCells);
            break;
          }
          default: {
          }
          }

          // If it's a requires, copy the data on over.  If it's a computes,
          // leave it as allocated but unused space.
          if (it->second.m_dep->m_dep_type == Task::Requires) {
            if (device_ptr == nullptr) {
              std::cerr << "ERROR: GPU variable's device pointer was nullptr"
                        << std::endl;
              throw ProblemSetupException(
                  "ERROR: GPU variable's device pointer was nullptr", __FILE__,
                  __LINE__);
            }

            if (it->second.m_dest == GpuUtilities::sameDeviceSameMpiRank) {

              // See if we get to be the thread that performs the H2D copy.

              // Figure out which thread gets to copy data H2D.  First touch
              // wins.  In case of a superpatch, the patch vars were shallow
              // copied so they all patches in the superpatch refer to the same
              // atomic status.
              bool performCopy = false;
              if (!staging) {
                performCopy = gpudw->compareAndSwapCopyingIntoGPU(
                  label_cstr, patchID, matlIndx, levelID);
              } else {
                performCopy = gpudw->compareAndSwapCopyingIntoGPUStaging(
                    label_cstr, patchID, matlIndx, levelID,
                    make_int3(low.x(), low.y(), low.z()),
                    make_int3(size.x(), size.y(), size.z()));
              }

              if (performCopy) {
                // This thread is doing the H2D copy for this simulation
                // variable.

                // Start by getting the host pointer.
                void *host_ptr = nullptr;

                // The variable exists in host memory.  We just have to get one
                // and copy it on in.
                switch (type) {
                case TypeDescription::CCVariable:
                case TypeDescription::NCVariable:
                case TypeDescription::SFCXVariable:
                case TypeDescription::SFCYVariable:
                case TypeDescription::SFCZVariable: {

                  // The var on the host could either be a regular var or a
                  // foreign var.
                  //  If it's a regular var, this will manage ghost cells by
                  //  creating a host var, rewindowing it, then copying in the
                  //  regions needed for the ghost cells. If this is the case,
                  //  then ghost cells for this specific instance of this var is
                  //  completed. If it's a foreign var, then there is no API at
                  //  the moment to query it directly (if you try to getGridVar
                  //  a foreign var, it doesn't work, it wasn't designed for
                  //  that).  Fortunately we would have already seen it in
                  //  whatever function called this. So use that instead.

                  // Note: Unhandled scenario:  If the adjacent patch is only in
                  // the GPU, this code doesn't gather it.
                  if (uses_SHRT_MAX) {
                    g_GridVarSuperPatch_mutex.lock();
                    {
                      // The variable wants the entire domain.  So we do a
                      // getRegion call instead.
                      GridVariableBase *gridVar =
                          dynamic_cast<GridVariableBase *>(
                              type_description->createInstance());

                      // dw->allocateAndPut(*gridVar, label, matlIndx, patch,
                      // ghosttype, numGhostCells, true);
                      if (!dw->exists(label, matlIndx, patch->getLevel())) {
                        // This creates and deep copies a region from the
                        // OnDemandDatawarehouse. It does so by deep copying
                        // from the other patches and forming one large region.
                        dw->getRegionModifiable(*gridVar, label, matlIndx,
                                                patch->getLevel(), low, high,
                                                true);
                        // passing in a clone (really it's just a shallow copy)
                        // to increase the reference counter by one
                        dw->putLevelDB(gridVar->clone(), label,
                                       patch->getLevel(), matlIndx);
                        // dw->getLevel(*constGridVar, label, matlIndx,
                        // patch->getLevel());
                      } else {
                        exit(-1);
                      }
                      // get the host pointer as well
                      host_ptr = gridVar->getBasePointer();

                      // let go of our reference, allowing a single reference to
                      // remain and keep the variable alive in leveDB. delete
                      // gridVar;
                      // TODO: Verify this cleans up.  If so change the comment.
                    }
                    g_GridVarSuperPatch_mutex.unlock();
                  } else {
                    if (it->second.m_var) {
                      // It's a foreign var.  We can't look it up, but we saw it
                      // previously.
                      GridVariableBase *gridVar =
                          dynamic_cast<GridVariableBase *>(it->second.m_var);
                      host_ptr = gridVar->getBasePointer();
                      // Since we didn't do a getGridVar() call, no reference to
                      // clean up
                    } else {
                      // I'm commenting carefully because this section has bit
                      // me several times.  If it's not done right, the bugs are
                      // a major headache to track down.  -- Brad P. Nov 30,
                      // 2016 We need all the data in the patch.  Perform a
                      // getGridVar(), which will return a var with the same
                      // window/data as the OnDemand DW variable, or it will
                      // create a new window/data sized to hold the room of the
                      // ghost cells and copy it into the gridVar variable.
                      // Internally it keeps track of refcounts for the window
                      // object and the data object. In any scenario treat the
                      // gridVar as a temporary copy of the actual var in the
                      // OnDemand DW, and as such that temporary variable needs
                      // to be reclaimed so there are no memory leaks.  The
                      // problem is that we need this temporary variable to live
                      // long enough to perform a device-to-host copy.
                      //* In one scenario with no ghost cells, you get back the
                      // same window/data just with refcounts incremented by 1.
                      //* In another scenario with ghost cells, the ref counts
                      // are at least 2, so deleting the gridVar won't
                      // automatically deallocate it
                      //* In another scenario with ghost cells, you get back a
                      // gridvar holding different window/data, their refcounts
                      // are 1
                      //   and so so deleting the gridVar will invoke
                      //   deallocation.  That would be bad if an async
                      //   device-to-host copy is needed.
                      // In all scenarios, the correct approach is just to delay
                      // deleting the gridVar object, and letting it persist
                      // until the all variable copies complete, then delete the
                      // object, which in turn decrements the refcounter, which
                      // then allows it to clean up later where needed (either
                      // immediately if the temp's refcounts hit 0, or later
                      // when the it does the scrub checks).

                      GridVariableBase *gridVar =
                          dynamic_cast<GridVariableBase *>(
                              type_description->createInstance());
                      dw->getGridVar(*gridVar, label, matlIndx, patch,
                                     ghosttype, numGhostCells);
                      host_ptr = gridVar->getBasePointer();
                      it->second.m_tempVarToReclaim =
                          gridVar; // This will be held onto so it persists, and
                                   // then cleaned up after the device-to-host
                                   // copy
                    }
                  }
                  break;
                }
                case TypeDescription::PerPatch: {
                  PerPatchBase *patchVar = dynamic_cast<PerPatchBase *>(
                      type_description->createInstance());
                  dw->get(*patchVar, label, matlIndx, patch);
                  host_ptr = patchVar->getBasePointer();
                  // let go of our reference
                  delete patchVar;
                  break;
                }
                case TypeDescription::ReductionVariable: {
                  ReductionVariableBase *reductionVar =
                      dynamic_cast<ReductionVariableBase *>(
                          type_description->createInstance());
                  dw->get(*reductionVar, label, patch->getLevel(), matlIndx);
                  host_ptr = reductionVar->getBasePointer();
                  // let go of our reference
                  delete reductionVar;
                  break;
                }
                default: {
                }
                }

                if (host_ptr != nullptr && device_ptr != nullptr) {
                  // Perform the copy!
                  gpuStream_t* stream =
                      dtask->getGpuStreamForThisTask(whichGPU);
                  OnDemandDataWarehouse::uintahSetGpuDevice(whichGPU);
                  if (it->second.m_varMemSize == 0) {
                    SCI_THROW(InternalError("Attempting to copy zero bytes to "
                                            "the GPU.  That shouldn't happen.",
                                            __FILE__, __LINE__));
                  }

                  // ABB 05/09/22: TODO: wait() needs to be removed for Async
                  // behaviour
                  stream->memcpy(device_ptr, host_ptr, it->second.m_varMemSize)
                      .wait();

                  // Tell this task that we're managing the copies for this
                  // variable.

                  dtask->getVarsBeingCopiedByTask().getMap().insert(
                      std::pair<GpuUtilities::LabelPatchMatlLevelDw,
                                DeviceGridVariableInfo>(it->first, it->second));
                }
              }
            } else if (it->second.m_dest ==
                           GpuUtilities::anotherDeviceSameMpiRank ||
                       it->second.m_dest == GpuUtilities::anotherMpiRank) {
              // We're not performing a host to GPU copy.  This is just prepare
              // a staging var. So it is a a gpu normal var to gpu staging var
              // copy. It is to prepare for upcoming GPU to host (MPI) or GPU to
              // GPU copies. Tell this task that we're managing the copies for
              // this variable.
              dtask->getVarsBeingCopiedByTask().getMap().insert(
                  std::pair<GpuUtilities::LabelPatchMatlLevelDw,
                            DeviceGridVariableInfo>(it->first, it->second));
            }
          }
        }
      }
    }
    isStaging = !isStaging;
  }
  //} end for (std::set<unsigned int>::const_iterator deviceNums_it =
  // deviceNums.begin() - this is commented out for now until multi-device
  // support is added
}

//______________________________________________________________________
//
void SYCLScheduler::prepareTaskVarsIntoTaskDW(DetailedTask *dtask) {
  // Copy all task variables metadata into the Task GPU DW.
  // All necessary metadata information must already exist in the host-side GPU
  // DWs.

  std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>
      &taskVarMap = dtask->getTaskVars().getMap();

  // Because maps are unordered, it is possible a staging var could be inserted
  // before the regular var exists. So just loop twice, once when all staging is
  // false, then loop again when all staging is true
  bool isStaging = false;

  for (int i = 0; i < 2; i++) {
    for (const auto it : taskVarMap) {
      // If isStaging is false, do the non-staging vars, then if isStaging is
      // true, do the staging vars. isStaging is flipped after the first
      // iteration of the i for loop.
      if (it->second.m_staging == isStaging) {
        switch (it->second.m_dep->m_var->typeDescription()->getType()) {
        case TypeDescription::PerPatch:
        case TypeDescription::ReductionVariable:
        case TypeDescription::CCVariable:
        case TypeDescription::NCVariable:
        case TypeDescription::SFCXVariable:
        case TypeDescription::SFCYVariable:
        case TypeDescription::SFCZVariable: {

          int dwIndex = it->second.m_dep->mapDataWarehouse();
          GPUDataWarehouse *gpudw =
              m_dws[dwIndex]->getGPUDW(it->second.m_whichGPU);
          int patchID = it->first.m_patchID;
          int matlIndx = it->first.m_matlIndx;
          int levelIndx = it->first.m_levelIndx;

          sycl::int3 offset{0,0,0};
          sycl::int3 size{0,0,0};
          if (it->second.m_staging) {
            offset = make_int3(it->second.m_offset.x(), it->second.m_offset.y(),
                               it->second.m_offset.z());
            size = make_int3(it->second.m_sizeVector.x(),
                             it->second.m_sizeVector.y(),
                             it->second.m_sizeVector.z());
          }

          GPUDataWarehouse *taskgpudw = dtask->getTaskGpuDataWarehouse(
              it->second.m_whichGPU, (Task::WhichDW)dwIndex);
          if (taskgpudw) {
            taskgpudw->copyItemIntoTaskDW(
                gpudw, it->second.m_dep->m_var->getName().c_str(), patchID,
                matlIndx, levelIndx, it->second.m_staging, offset, size);
          } else {
            SCI_THROW(InternalError("No task data warehouse found\n", __FILE__,
                                    __LINE__));
          }
        } break;
        default: {
        }
        }
      }
    }
    isStaging = !isStaging;
  }
}

//______________________________________________________________________
//
void SYCLScheduler::prepareGhostCellsIntoTaskDW(DetailedTask *dtask) {

  // Tell the Task DWs about any ghost cells they will need to process.
  // This adds in entries into the task DW's d_varDB which isn't a var, but is
  // instead metadata describing how to copy ghost cells between two vars listed
  // in d_varDB.

  const std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo>
      &ghostVarMap = dtask->getGhostVars().getMap();
  for (const auto it = ghostVarMap) {
    // If the neighbor is valid on the GPU, we just send in from and to
    // coordinates and call a kernel to copy those coordinates If it's not valid
    // on the GPU, we copy in the grid var and send in from and to coordinates
    // and call a kernel to copy those coordinates.

    // Peer to peer GPU copies will be handled elsewhere.
    // GPU to another MPI ranks will be handled elsewhere.
    if (it->second.m_dest != GpuUtilities::anotherDeviceSameMpiRank &&
        it->second.m_dest != GpuUtilities::anotherMpiRank) {
      int dwIndex = it->first.m_dataWarehouse;

      // We can copy it manually internally within the device via a kernel.
      // This apparently goes faster overall
      IntVector varOffset = it->second.m_varOffset;
      IntVector varSize = it->second.m_varSize;
      IntVector ghost_low = it->first.m_sharedLowCoordinates;
      IntVector ghost_high = it->first.m_sharedHighCoordinates;
      IntVector virtualOffset = it->second.m_virtualOffset;

      // Add in an entry into this Task DW's d_varDB which isn't a var, but is
      // instead metadata describing how to copy ghost cells between two vars
      // listed in d_varDB.
      dtask
          ->getTaskGpuDataWarehouse(it->second.m_sourceDeviceNum,
                                    (Task::WhichDW)dwIndex)
          ->putGhostCell(
              it->first.m_label.c_str(),
              it->second.m_sourcePatchPointer->getID(),
              it->second.m_destPatchPointer->getID(), it->first.m_matlIndx,
              it->first.m_levelIndx, it->second.m_sourceStaging,
              it->second.m_destStaging,
              make_int3(varOffset.x(), varOffset.y(), varOffset.z()),
              make_int3(varSize.x(), varSize.y(), varSize.z()),
              make_int3(ghost_low.x(), ghost_low.y(), ghost_low.z()),
              make_int3(ghost_high.x(), ghost_high.y(), ghost_high.z()),
              make_int3(virtualOffset.x(), virtualOffset.y(),
                        virtualOffset.z()));
    }
  }
}

//______________________________________________________________________
//
bool SYCLScheduler::ghostCellsProcessingReady(DetailedTask *dtask) {

  // Check that all staging data is ready for upcoming GPU to GPU ghost cell
  // copies.  Most of the time, the staging data is "inside" the patch variable
  // in the data warehouse.  But sometimes, some upcoming GPU to GPU ghost cell
  // copies will occur from other patches not assigned to the task. For example,
  // ghost cell data from another MPI rank is going to be assigned to a
  // different patch, but the ghost cell data needs to be copied into the patch
  // variable assigned to this task. In another example, ghost cell data may be
  // coming from another patch that is on the same MPI rank, but that other
  // patch variable is being copied host-to-GPU by another scheduler thread
  // processing another task. The best solution here is to investigate all
  // upcoming ghost cell copies that need to occur, and verify that both the
  // source and destination patches are valid in the memory space. Note: I bet
  // this is more accurate than the above check, if so, remove the above loop.

  const std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo>
      &ghostVarMap = dtask->getGhostVars().getMap();
  for (const auto it : ghostVarMap) {

    GPUDataWarehouse *gpudw = m_dws[it->first.m_dataWarehouse]->getGPUDW(
        GpuUtilities::getGpuIndexForPatch(it->second.m_sourcePatchPointer));

    // Check the source
    if (it->second.m_sourceStaging) {
      if (!(gpudw->areAllStagingVarsValid(
              it->first.m_label.c_str(),
              it->second.m_sourcePatchPointer->getID(), it->second.m_matlIndx,
              it->first.m_levelIndx))) {
        return false;
      }
    } else {
      if (!(gpudw->isValidOnGPU(it->first.m_label.c_str(),
                                it->second.m_sourcePatchPointer->getID(),
                                it->second.m_matlIndx,
                                it->first.m_levelIndx))) {
        return false;
      }
    }

    // Check the destination?
  }

  // if we got there, then everything must be ready to go.
  return true;
}

//______________________________________________________________________
//
bool SYCLScheduler::allHostVarsProcessingReady(DetailedTask *dtask) {

  const Task *task = dtask->getTask();

  dtask->clearPreparationCollections();

  // Gather up all possible dependents from requires and computes and remove
  // duplicates (we don't want to transfer some variables twice). Note: A task
  // can only run on one level at a time.  It could run multiple patches and
  // multiple materials, but a single task will never run multiple levels.
  std::map<labelPatchMatlDependency, const Task::Dependency *> vars;
  for (const Task::Dependency *dependantVar = task->getRequires();
       dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches =
        dependantVar->getPatchesUnderDomain(dtask->getPatches());
    if (patches) {
      constHandle<MaterialSubset> matls =
          dependantVar->getMaterialsUnderDomain(dtask->getMaterials());
      const int numPatches = patches->size();
      const int numMatls = matls->size();
      for (int i = 0; i < numPatches; i++) {
        for (int j = 0; j < numMatls; j++) {
          labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(),
                                        patches->get(i)->getID(), matls->get(j),
                                        Task::Requires);
          if (vars.find(lpmd) == vars.end()) {
            vars.insert(
                std::map<labelPatchMatlDependency,
                         const Task::Dependency *>::value_type(lpmd,
                                                               dependantVar));
          }
        }
      }
    } else {
      std::cout << myRankThread()
                << " In allHostVarsProcessingReady, no patches, task is "
                << dtask->getName() << std::endl;
    }
  }

  // Go through each var, see if it's valid or valid with ghosts.
  for (auto varIter : vars) {
    const Task::Dependency *curDependency = varIter->second;

    constHandle<PatchSubset> patches =
        curDependency->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls =
        curDependency->getMaterialsUnderDomain(dtask->getMaterials());
    const int numPatches = patches->size();
    const int patchID = varIter->first.m_patchID;
    const Patch *patch = nullptr;
    const Level *level = nullptr;
    for (int i = 0; i < numPatches; i++) {
      if (patches->get(i)->getID() == patchID) {
        patch = patches->get(i);
        level = patch->getLevel();
      }
    }
    int levelID = level->getID();
    if (curDependency->m_var->typeDescription()->getType() ==
        TypeDescription::ReductionVariable) {
      levelID = -1;
    }
    const int matlID = varIter->first.m_matlIndex;
    const int dwIndex = curDependency->mapDataWarehouse();
    OnDemandDataWarehouseP dw = m_dws[dwIndex];
    GPUDataWarehouse *gpudw =
        dw->getGPUDW(GpuUtilities::getGpuIndexForPatch(patch));
    if (curDependency->m_dep_type == Task::Requires) {
      if (gpudw->dwEntryExistsOnCPU(curDependency->m_var->getName().c_str(),
                                    patchID, matlID, levelID)) {
        if (!(gpudw->isValidOnCPU(curDependency->m_var->getName().c_str(),
                                  patchID, matlID, levelID))) {
          return false;
        }
      }
    }
  }

  // if we got there, then everything must be ready to go.
  return true;
}

//______________________________________________________________________
//
bool SYCLScheduler::allGPUVarsProcessingReady(DetailedTask *dtask) {

  const Task *task = dtask->getTask();

  dtask->clearPreparationCollections();

  // Gather up all possible dependents from requires and computes and remove
  // duplicates (we don't want to transfer some variables twice). Note: A task
  // can only run on one level at a time.  It could run multiple patches and
  // multiple materials, but a single task will never run multiple levels.
  std::map<labelPatchMatlDependency, const Task::Dependency *> vars;
  for (const Task::Dependency *dependantVar = task->getRequires();
       dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches =
        dependantVar->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls =
        dependantVar->getMaterialsUnderDomain(dtask->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(),
                                      patches->get(i)->getID(), matls->get(j),
                                      Task::Requires);
        if (vars.find(lpmd) == vars.end()) {
          vars.insert(
              std::map<labelPatchMatlDependency,
                       const Task::Dependency *>::value_type(lpmd,
                                                             dependantVar));
        }
      }
    }
  }

  // Go through each var, see if it's valid or valid with ghosts.
  for (auto varIter : vars) {
    const Task::Dependency *curDependency = varIter->second;

    constHandle<PatchSubset> patches =
        curDependency->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls =
        curDependency->getMaterialsUnderDomain(dtask->getMaterials());
    const int numPatches = patches->size();
    const int patchID = varIter->first.m_patchID;
    const Patch *patch = nullptr;
    const Level *level = nullptr;
    for (int i = 0; i < numPatches; i++) {
      if (patches->get(i)->getID() == patchID) {
        patch = patches->get(i);
        level = patch->getLevel();
      }
    }
    int levelID = level->getID();
    if (curDependency->m_var->typeDescription()->getType() ==
        TypeDescription::ReductionVariable) {
      levelID = -1;
    }

    const int matlID = varIter->first.m_matlIndex;
    const int dwIndex = curDependency->mapDataWarehouse();
    OnDemandDataWarehouseP dw = m_dws[dwIndex];
    GPUDataWarehouse *gpudw =
        dw->getGPUDW(GpuUtilities::getGpuIndexForPatch(patch));
    if (curDependency->m_dep_type == Task::Requires) {
      if (curDependency->m_gtype != Ghost::None &&
          curDependency->m_num_ghost_cells > 0) {
        // it has ghost cells.
        if (!(gpudw->isValidWithGhostsOnGPU(
                curDependency->m_var->getName().c_str(), patchID, matlID,
                levelID))) {
          return false;
        } else {
        }
      } else {
        // If it's a gridvar, then we just don't have the ghost cells processed
        // yet by another thread If it's another type of variable, something
        // went wrong, it should have been marked as valid previously.
        if (!(gpudw->isValidOnGPU(curDependency->m_var->getName().c_str(),
                                  patchID, matlID, levelID))) {
          return false;
        } else {
        }
      }
    }
  }

  // if we got there, then everything must be ready to go.
  return true;
}

//______________________________________________________________________
//
void SYCLScheduler::markDeviceRequiresDataAsValid(DetailedTask *dtask) {

  // This marks any Requires variable as valid that wasn't in the GPU but is now
  // in the GPU. If they were already in the GPU due to being computes from a
  // previous time step, it was already marked as valid.  So there is no need to
  // do anything extra for them. If they weren't in the GPU yet, this task or
  // another task copied it in. If it's another task that copied it in, we let
  // that task manage it. If it was this task, then those variables which this
  // task copied in are found in varsBeingCopiedByTask. By the conclusion of
  // this method, some variables will be valid and awaiting ghost cells, some
  // will just be valid if they had no ghost cells, and some variables will be
  // undetermined if they're being managed by another task. After this method, a
  // kernel is invoked to process ghost cells.

  // Go through device requires vars and mark them as valid on the device.  They
  // are either already valid because they were there previously.  Or they just
  // got copied in and the stream completed.
  std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>
      &varMap = dtask->getVarsBeingCopiedByTask().getMap();
  for (auto it : varMap) {
    int whichGPU = it->second.m_whichGPU;
    int dwIndex = it->second.m_dep->mapDataWarehouse();
    GPUDataWarehouse *gpudw = m_dws[dwIndex]->getGPUDW(whichGPU);
    if (it->second.m_dep->m_dep_type == Task::Requires) {
      if (!it->second.m_staging) {
        gpudw->compareAndSwapSetValidOnGPU(
            it->second.m_dep->m_var->getName().c_str(), it->first.m_patchID,
            it->first.m_matlIndx, it->first.m_levelIndx);
      } else {
        gpudw->compareAndSwapSetValidOnGPUStaging(
            it->second.m_dep->m_var->getName().c_str(), it->first.m_patchID,
            it->first.m_matlIndx, it->first.m_levelIndx,
            make_int3(it->second.m_offset.x(), it->second.m_offset.y(),
                      it->second.m_offset.z()),
            make_int3(it->second.m_sizeVector.x(), it->second.m_sizeVector.y(),
                      it->second.m_sizeVector.z()));
      }

      if (it->second.m_tempVarToReclaim) {
        // Release our reference to the variable data that getGridVar returned
        delete it->second.m_tempVarToReclaim;
      }
    }
  }
}

//______________________________________________________________________
//
void SYCLScheduler::markDeviceGhostsAsValid(DetailedTask *dtask) {
  // Go through requires vars and mark them as valid on the device.  They are
  // either already valid because they were there previously.  Or they just got
  // copied in and the stream completed. Now go through the varsToBeGhostReady
  // collection.  Any in there should be marked as valid with ghost cells
  std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>
      &varMap = dtask->getVarsToBeGhostReady().getMap();
  for (auto it : varMap) {
    int whichGPU = it->second.m_whichGPU;
    int dwIndex = it->second.m_dep->mapDataWarehouse();
    GPUDataWarehouse *gpudw = m_dws[dwIndex]->getGPUDW(whichGPU);

    gpudw->setValidWithGhostsOnGPU(it->second.m_dep->m_var->getName().c_str(),
                                   it->first.m_patchID, it->first.m_matlIndx,
                                   it->first.m_levelIndx);
  }
}

//______________________________________________________________________
//
void SYCLScheduler::markDeviceComputesDataAsValid(DetailedTask *dtask) {
  // Go through device computes vars and mark them as valid on the device.

  // The only thing we need to process is the requires.
  const Task *task = dtask->getTask();
  for (const Task::Dependency *comp = task->getComputes(); comp != 0;
       comp = comp->m_next) {
    constHandle<PatchSubset> patches =
        comp->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls =
        comp->getMaterialsUnderDomain(dtask->getMaterials());
    // this is so we can allocate persistent events and streams to distribute
    // when needed one stream and one event per variable per H2D copy
    // (numPatches * numMatls)
    int numPatches = patches->size();
    int numMatls = matls->size();
    int dwIndex = comp->mapDataWarehouse();
    OnDemandDataWarehouseP dw = m_dws[dwIndex];

    for (int i = 0; i < numPatches; i++) {
      GPUDataWarehouse *gpudw =
          dw->getGPUDW(GpuUtilities::getGpuIndexForPatch(patches->get(i)));
      if (gpudw != nullptr) {
        for (int j = 0; j < numMatls; j++) {
          int patchID = patches->get(i)->getID();
          int matlID = matls->get(j);
          const Level *level = patches->get(i)->getLevel();
          int levelID = level->getID();
          if (gpudw->isAllocatedOnGPU(comp->m_var->getName().c_str(), patchID,
                                      matlID, levelID)) {
            gpudw->compareAndSwapSetValidOnGPU(comp->m_var->getName().c_str(),
                                               patchID, matlID, levelID);
          }
        }
      }
    }
  }
}

//______________________________________________________________________
//
void SYCLScheduler::markHostRequiresDataAsValid(DetailedTask *dtask) {
  // Data has been copied from the device to the host.  The stream has
  // completed. Go through all variables that this CPU task was responsible for
  // copying mark them as valid on the CPU

  // The only thing we need to process is the requires.
  std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>
      &varMap = dtask->getVarsBeingCopiedByTask().getMap();
  for (auto it : varMap) {
    int whichGPU = it->second.m_whichGPU;
    int dwIndex = it->second.m_dep->mapDataWarehouse();
    GPUDataWarehouse *gpudw = m_dws[dwIndex]->getGPUDW(whichGPU);
    if (it->second.m_dep->m_dep_type == Task::Requires) {
      if (!it->second.m_staging) {
        gpudw->compareAndSwapSetValidOnCPU(
            it->second.m_dep->m_var->getName().c_str(), it->first.m_patchID,
            it->first.m_matlIndx, it->first.m_levelIndx);
      }
      if (it->second.m_var) {
        // Release our reference to the variable data that getGridVar returned
        delete it->second.m_var;
      }
    }
  }
}

//______________________________________________________________________
//
void SYCLScheduler::initiateD2HForHugeGhostCells(DetailedTask *dtask) {
  // RMCRT problems use 32768 ghost cells as a way to force an "all to all"
  // transmission of ghost cells It is much easier to manage these ghost cells
  // in host memory instead of GPU memory.  So for such variables, after they
  // are done computing, we will copy them D2H.  For RMCRT, this overhead only
  // adds about 1% or less to the overall computation time.  ' This only works
  // with COMPUTES, it is not configured to work with requires.

  const Task *task = dtask->getTask();

  // determine which computes variables to copy back to the host
  for (const Task::Dependency *comp = task->getComputes(); comp != 0;
       comp = comp->m_next) {
    // Only process large number of ghost cells.
    if (comp->m_num_ghost_cells == SHRT_MAX) {
      constHandle<PatchSubset> patches =
          comp->getPatchesUnderDomain(dtask->getPatches());
      constHandle<MaterialSubset> matls =
          comp->getMaterialsUnderDomain(dtask->getMaterials());

      int dwIndex = comp->mapDataWarehouse();
      OnDemandDataWarehouseP dw = m_dws[dwIndex];

      void *host_ptr = nullptr;   // host base pointer to raw data
      void *device_ptr = nullptr; // device base pointer to raw data
      std::size_t host_bytes = 0; // raw byte count to copy to the device

      sycl::int3 host_low, host_high, host_offset, host_size, host_strides;

      int numPatches = patches->size();
      int numMatls = matls->size();
      //__________________________________
      //
      for (int i = 0; i < numPatches; ++i) {
        for (int j = 0; j < numMatls; ++j) {
          const int patchID = patches->get(i)->getID();
          const int matlID = matls->get(j);

          const std::string compVarName = comp->m_var->getName();

          const Patch *patch = nullptr;
          const Level *level = nullptr;
          for (int i = 0; i < numPatches; i++) {
            if (patches->get(i)->getID() == patchID) {
              patch = patches->get(i);
              level = patch->getLevel();
            }
          }
          if (!patch) {
            SCI_THROW(
                InternalError("SYCLScheduler::initiateD2HForHugeGhostCells()"
                              " patch not found.",
                              __FILE__, __LINE__));
          }
          const int levelID = level->getID();

          const unsigned int deviceNum =
              GpuUtilities::getGpuIndexForPatch(patch);
          GPUDataWarehouse *gpudw = dw->getGPUDW(deviceNum);
          OnDemandDataWarehouse::uintahSetGpuDevice(deviceNum);
          gpuStream_t *stream = dtask->getGpuStreamForThisTask(deviceNum);

          if (gpudw != nullptr) {

            // It's not valid on the CPU but it is on the GPU.  Copy it on over.
            if (!gpudw->isValidOnCPU(compVarName.c_str(), patchID, matlID,
                                     levelID)) {
              const TypeDescription::Type type =
                  comp->m_var->typeDescription()->getType();
              const TypeDescription::Type datatype =
                  comp->m_var->typeDescription()->getSubType()->getType();
              switch (type) {
              case TypeDescription::CCVariable:
              case TypeDescription::NCVariable:
              case TypeDescription::SFCXVariable:
              case TypeDescription::SFCYVariable:
              case TypeDescription::SFCZVariable: {

                bool performCopy = gpudw->compareAndSwapCopyingIntoCPU(
                    compVarName.c_str(), patchID, matlID, levelID);
                if (performCopy) {
                  // size the host var to be able to fit all r::oom needed.
                  IntVector host_low, host_high, host_lowOffset,
                      host_highOffset, host_offset, host_size, host_strides;
                  level->computeVariableExtents(type, host_low, host_high);
                  int dwIndex = comp->mapDataWarehouse();
                  OnDemandDataWarehouseP dw = m_dws[dwIndex];

                  // It's possible the computes data may contain ghost cells.
                  // But a task needing to get the data out of the GPU may not
                  // know this.  It may just want the var data. This creates a
                  // dilemma, as the GPU var is sized differently than the CPU
                  // var. So ask the GPU what size it has for the var.  Size the
                  // CPU var to match so it can copy all GPU data in. When the
                  // GPU->CPU copy is done, then we need to resize the CPU var
                  // if needed to match what the CPU is expecting it to be.
                  // GPUGridVariableBase* gpuGridVar;

                  sycl::int3 low;
                  sycl::int3 high;
                  sycl::int3 size;
                  GPUDataWarehouse::GhostType tempgtype;
                  Ghost::GhostType gtype;
                  int numGhostCells;
                  gpudw->getSizes(low, high, size, tempgtype, numGhostCells,
                                  compVarName.c_str(), patchID, matlID,
                                  levelID);

                  gtype = (Ghost::GhostType)tempgtype;

                  GridVariableBase *gridVar = dynamic_cast<GridVariableBase *>(
                      comp->m_var->typeDescription()->createInstance());

                  bool finalized = dw->isFinalized();
                  if (finalized) {
                    dw->unfinalize();
                  }

                  dw->allocateAndPut(*gridVar, comp->m_var, matlID, patch,
                                     gtype, numGhostCells);
                  if (finalized) {
                    dw->refinalize();
                  }

                  gridVar->getSizes(host_low, host_high, host_offset, host_size,
                                    host_strides);
                  host_ptr = gridVar->getBasePointer();
                  host_bytes = gridVar->getDataSize();

                  sycl::int3 device_size{0, 0, 0};
                  sycl::int3 device_offset{0, 0, 0};
                  GPUGridVariableBase *device_var =
                      OnDemandDataWarehouse::createGPUGridVariable(datatype);
                  gpudw->get(*device_var, compVarName.c_str(), patchID, matlID,
                             levelID);
                  device_var->getArray3(device_offset, device_size, device_ptr);
                  delete device_var;

                  // if offset and size is equal to CPU DW, directly copy back
                  // to CPU var memory;
                  if (device_offset.x() == host_low.x() &&
                      device_offset.y() == host_low.y() &&
                      device_offset.z() == host_low.z() &&
                      device_size.x() == host_size.x() &&
                      device_size.y() == host_size.y() &&
                      device_size.z() == host_size.z()) {

                    stream->memcpy(host_ptr, device_ptr, host_bytes).wait();

                    dtask->getVarsBeingCopiedByTask().add(
                        patch, matlID, levelID, false,
                        IntVector(device_size.x(), device_size.y(),
                                  device_size.z()),
                        host_strides.x(), host_bytes,
                        IntVector(device_offset.x(), device_offset.y(),
                                  device_offset.z()),
                        comp, gtype, numGhostCells, deviceNum, gridVar,
                        GpuUtilities::sameDeviceSameMpiRank);
                  }
                  delete gridVar;
                }
                break;
              }
              default:
                std::ostringstream warn;
                warn << "  ERROR: "
                        "SYCLScheduler::initiateD2HForHugeGhostCells ("
                     << dtask->getName()
                     << ") variable: " << comp->m_var->getName()
                     << " not implemented " << std::endl;
                SCI_THROW(InternalError(warn.str(), __FILE__, __LINE__));
              }
            }
          }
        }
      }
    }
  }
}

//______________________________________________________________________
//
void SYCLScheduler::initiateD2H(DetailedTask *dtask) {
  // Request that all contiguous device arrays from the device be sent to their
  // contiguous host array counterparts. We only copy back the data needed for
  // an upcoming task.  If data isn't needed, it can stay on the device and
  // potentially even die on the device

  // Returns true if no device data is required, thus allowing a CPU task to
  // immediately proceed.

  void *host_ptr = nullptr;   // host base pointer to raw data
  void *device_ptr = nullptr; // device base pointer to raw data
  std::size_t host_bytes = 0; // raw byte count to copy to the device

  const Task *task = dtask->getTask();
  dtask->clearPreparationCollections();

  // The only thing we need to process is the requires.
  // Gather up all possible dependents and remove duplicate (we don't want to
  // transfer some variables twice)
  std::map<labelPatchMatlDependency, const Task::Dependency *> vars;
  for (const Task::Dependency *dependantVar = task->getRequires();
       dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches =
        dependantVar->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls =
        dependantVar->getMaterialsUnderDomain(dtask->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(),
                                      patches->get(i)->getID(), matls->get(j),
                                      Task::Requires);
        if (vars.find(lpmd) == vars.end()) {
          vars.insert(
              std::map<labelPatchMatlDependency,
                       const Task::Dependency *>::value_type(lpmd,
                                                             dependantVar));
        }
      }
    }
  }

  for (const Task::Dependency *dependantVar = task->getComputes();
       dependantVar != 0; dependantVar = dependantVar->m_next) {
    constHandle<PatchSubset> patches =
        dependantVar->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls =
        dependantVar->getMaterialsUnderDomain(dtask->getMaterials());
    const int numPatches = patches->size();
    const int numMatls = matls->size();
    for (int i = 0; i < numPatches; i++) {
      for (int j = 0; j < numMatls; j++) {
        labelPatchMatlDependency lpmd(dependantVar->m_var->getName().c_str(),
                                      patches->get(i)->getID(), matls->get(j),
                                      Task::Computes);
        if (vars.find(lpmd) == vars.end()) {
        }
      }
    }
  }

  // Go through each unique dependent var and see if we should queue up a D2H
  // copy
  for (auto varIter : vars) {
    const Task::Dependency *dependantVar = varIter->second;
    constHandle<PatchSubset> patches =
        dependantVar->getPatchesUnderDomain(dtask->getPatches());
    constHandle<MaterialSubset> matls =
        dependantVar->getMaterialsUnderDomain(dtask->getMaterials());

    // this is so we can allocate persistent events and streams to distribute
    // when needed
    //   one stream and one event per variable per H2D copy (numPatches *
    //   numMatls)

    int numPatches = patches->size();
    int dwIndex = dependantVar->mapDataWarehouse();
    OnDemandDataWarehouseP dw = m_dws[dwIndex];

    // Find the patch and level objects associated with the patchID
    const int patchID = varIter->first.m_patchID;
    const Patch *patch = nullptr;
    const Level *level = nullptr;
    for (int i = 0; i < numPatches; i++) {
      if (patches->get(i)->getID() == patchID) {
        patch = patches->get(i);
        level = patch->getLevel();
      }
    }

    if (!patch) {
      SCI_THROW(InternalError("SYCLScheduler::initiateD2H() patch not found.",
                              __FILE__, __LINE__));
    }

    int levelID = level->getID();
    if (dependantVar->m_var->typeDescription()->getType() ==
        TypeDescription::ReductionVariable) {
      levelID = -1;
    }

    const int matlID = varIter->first.m_matlIndex;

    unsigned int deviceNum = GpuUtilities::getGpuIndexForPatch(patch);
    GPUDataWarehouse *gpudw = dw->getGPUDW(deviceNum);
    OnDemandDataWarehouse::uintahSetGpuDevice(deviceNum);
    gpuStream_t *stream = dtask->getGpuStreamForThisTask(deviceNum);

    const std::string varName = dependantVar->m_var->getName();

    // TODO: Titan production hack.  A clean hack, but should be fixed. Brad P
    // Dec 1 2016 There currently exists a race condition.  Suppose cellType is
    // in both host and GPU memory.  Currently the GPU data warehouse knows it
    // is in GPU memory, but it doesn't know if it's in host memory (the GPU DW
    // doesn't track lifetimes of host DW vars). Thread 2 - Task A requests a
    // requires var for cellType for the host newDW, and gets it. Thread 3 -
    // Task B invokes the initiateD2H check, thinks there is no host instance of
    // cellType,
    //            so it initiates a D2H, which performs another host
    //            allocateAndPut, and the subsequent put deletes the old entry
    //            and creates a new entry.
    // Race condition is that thread 2's pointer has been cleaned up, while
    // thread 3 has a new one. A temp fix could be to check if all host vars
    // exist in the host dw prior to launching the task.

    // if (varName != "divQ" && varName != "RMCRTboundFlux" && varName !=
    // "radiationVolq" ) {
    //   continue;
    // }

    if (gpudw != nullptr) {
      // It's not valid on the CPU but it is on the GPU.  Copy it on over.
      if (!gpudw->isValidOnCPU(varName.c_str(), patchID, matlID, levelID) &&
          gpudw->isAllocatedOnGPU(varName.c_str(), patchID, matlID, levelID) &&
          gpudw->isValidOnGPU(varName.c_str(), patchID, matlID, levelID)) {

        const TypeDescription::Type type =
            dependantVar->m_var->typeDescription()->getType();
        const TypeDescription::Type datatype =
            dependantVar->m_var->typeDescription()->getSubType()->getType();
        switch (type) {
        case TypeDescription::CCVariable:
        case TypeDescription::NCVariable:
        case TypeDescription::SFCXVariable:
        case TypeDescription::SFCYVariable:
        case TypeDescription::SFCZVariable: {
          bool performCopy = gpudw->compareAndSwapCopyingIntoCPU(
              varName.c_str(), patchID, matlID, levelID);
          if (performCopy) {

            // It's possible the computes data may contain ghost cells.  But a
            // task needing to get the data out of the GPU may not know this. It
            // may just want the var data. This creates a dilemma, as the GPU
            // var is sized differently than the CPU var. So ask the GPU what
            // size it has for the var.  Size the CPU var to match so it can
            // copy all GPU data in. When the GPU->CPU copy is done, then we
            // need to resize the CPU var if needed to match what the CPU is
            // expecting it to be.

            // Get the host var variable
            GridVariableBase *gridVar = dynamic_cast<GridVariableBase *>(
                dependantVar->m_var->typeDescription()->createInstance());
            const std::size_t elementDataSize =
                OnDemandDataWarehouse::getTypeDescriptionSize(
                    dependantVar->m_var->typeDescription()
                        ->getSubType()
                        ->getType());

            // The device will have our best knowledge of the exact
            // dimensions/ghost cells of the variable, so lets get those values.
            sycl::int3 device_low;
            sycl::int3 device_offset;
            sycl::int3 device_high;
            sycl::int3 device_size;
            GPUDataWarehouse::GhostType tempgtype;
            Ghost::GhostType gtype;
            int numGhostCells;
            gpudw->getSizes(device_low, device_high, device_size, tempgtype,
                            numGhostCells, varName.c_str(), patchID, matlID,
                            levelID);
            gtype = (Ghost::GhostType)tempgtype;
            device_offset = device_low;

            // Now get dimensions for the host variable.
            bool uses_SHRT_MAX = (numGhostCells == SHRT_MAX);
            Patch::VariableBasis basis =
                Patch::translateTypeToBasis(type, false);

            // Get the size/offset of what the host var would be with ghost
            // cells.
            IntVector host_low, host_high, host_lowOffset, host_highOffset,
                host_offset, host_size;
            if (uses_SHRT_MAX) {
              level->findCellIndexRange(host_low,
                                        host_high); // including extraCells
            } else {
              Patch::getGhostOffsets(type, gtype, numGhostCells, host_lowOffset,
                                     host_highOffset);
              patch->computeExtents(
                  basis, dependantVar->m_var->getBoundaryLayer(),
                  host_lowOffset, host_highOffset, host_low, host_high);
            }
            host_size = host_high - host_low;
            int dwIndex = dependantVar->mapDataWarehouse();
            OnDemandDataWarehouseP dw = m_dws[dwIndex];

            // Get/make the host var

            // get the device var so we can get the pointer.
            GPUGridVariableBase *device_var =
                OnDemandDataWarehouse::createGPUGridVariable(datatype);
            gpudw->get(*device_var, varName.c_str(), patchID, matlID, levelID);
            device_var->getArray3(device_offset, device_size, device_ptr);
            delete device_var;

            bool proceedWithCopy = false;
            // See if the size of the host var and the device var match.
            if (device_offset.x() == host_low.x() &&
                device_offset.y() == host_low.y() &&
                device_offset.z() == host_low.z() &&
                device_size.x() == host_size.x() &&
                device_size.y() == host_size.y() &&
                device_size.z() == host_size.z()) {
              proceedWithCopy = true;

              // Note, race condition possible here
              bool finalized = dw->isFinalized();
              if (finalized) {
                dw->unfinalize();
              }
              if (uses_SHRT_MAX) {
                gridVar->allocate(host_low, host_high);
              } else {
                dw->allocateAndPut(*gridVar, dependantVar->m_var, matlID, patch,
                                   gtype, numGhostCells);
              }
              if (finalized) {
                dw->refinalize();
              }
            } else {
              // They didn't match.  Lets see if the device var doesn't have
              // ghost cells. This can happen prior to the first timestep during
              // initial computations when no variables had room for ghost
              // cells.
              Patch::getGhostOffsets(type, Ghost::None, 0, host_lowOffset,
                                     host_highOffset);
              patch->computeExtents(
                  basis, dependantVar->m_var->getBoundaryLayer(),
                  host_lowOffset, host_highOffset, host_low, host_high);

              host_size = host_high - host_low;
              if (device_offset.x() == host_low.x() &&
                  device_offset.y() == host_low.y() &&
                  device_offset.z() == host_low.z() &&
                  device_size.x() == host_size.x() &&
                  device_size.y() == host_size.y() &&
                  device_size.z() == host_size.z()) {

                proceedWithCopy = true;

                // Note, race condition possible here
                bool finalized = dw->isFinalized();
                if (finalized) {
                  dw->unfinalize();
                }
                dw->allocateAndPut(*gridVar, dependantVar->m_var, matlID, patch,
                                   Ghost::None, 0);
                if (finalized) {
                  dw->refinalize();
                }
              } else {
                // The sizes STILL don't match. One more last ditch effort.
                // Assume it was using up to 32768 ghost cells.
                level->findCellIndexRange(host_low, host_high);
                host_size = host_high - host_low;

                if (device_offset.x() == host_low.x() &&
                    device_offset.y() == host_low.y() &&
                    device_offset.z() == host_low.z() &&
                    device_size.x() == host_size.x() &&
                    device_size.y() == host_size.y() &&
                    device_size.z() == host_size.z()) {

                  // ok, this worked.  Allocate it the large ghost cell way with
                  // getRegion Note, race condition possible here
                  bool finalized = dw->isFinalized();
                  if (finalized) {
                    dw->unfinalize();
                  }
                  gridVar->allocate(host_low, host_high);
                  if (finalized) {
                    dw->refinalize();
                  }
                  proceedWithCopy = true;
                } else {
                  SCI_THROW(InternalError("SYCLScheduler::initiateD2H() - "
                                          "Device and host sizes didn't match.",
                                          __FILE__, __LINE__));
                }
              }
            }

            // if offset and size is equal to CPU DW, directly copy back to CPU
            // var memory;
            if (proceedWithCopy) {

              host_ptr = gridVar->getBasePointer();
              host_bytes = gridVar->getDataSize();

              stream->memcpy(host_ptr, device_ptr, host_bytes);

              dtask->getVarsBeingCopiedByTask().add(
                  patch, matlID, levelID, false,
                  IntVector(device_size.x(), device_size.y(), device_size.z()),
                  elementDataSize, host_bytes,
                  IntVector(device_offset.x(), device_offset.y(),
                            device_offset.z()),
                  dependantVar, gtype, numGhostCells, deviceNum, gridVar,
                  GpuUtilities::sameDeviceSameMpiRank);
            }
            // delete gridVar;
          }
          break;
        }
        case TypeDescription::PerPatch: {
          bool performCopy = gpudw->compareAndSwapCopyingIntoCPU(
              varName.c_str(), patchID, matlID, levelID);
          if (performCopy) {

            PerPatchBase *hostPerPatchVar = dynamic_cast<PerPatchBase *>(
                dependantVar->m_var->typeDescription()->createInstance());
            const bool finalized = dw->isFinalized();
            if (finalized) {
              dw->unfinalize();
            }
            dw->put(*hostPerPatchVar, dependantVar->m_var, matlID, patch);
            if (finalized) {
              dw->refinalize();
            }
            host_ptr = hostPerPatchVar->getBasePointer();
            host_bytes = hostPerPatchVar->getDataSize();

            GPUPerPatchBase *gpuPerPatchVar =
                OnDemandDataWarehouse::createGPUPerPatch(datatype);
            gpudw->get(*gpuPerPatchVar, varName.c_str(), patchID, matlID,
                       levelID);
            device_ptr = gpuPerPatchVar->getVoidPointer();
            std::size_t device_bytes = gpuPerPatchVar->getMemSize();
            delete gpuPerPatchVar;

            if (host_bytes == device_bytes) {
              stream->memcpy(host_ptr, device_ptr, host_bytes);
              dtask->getVarsBeingCopiedByTask().add(
                patch, matlID, levelID, host_bytes, host_bytes, dependantVar,
                deviceNum, hostPerPatchVar,
                GpuUtilities::sameDeviceSameMpiRank);
            } else {
              SCI_THROW(InternalError(
                          "InitiateD2H - PerPatch variable memory sizes didn't match",
                          __FILE__, __LINE__));
            }
            // delete hostPerPatchVar;
          }

          break;
        }
        case TypeDescription::ReductionVariable: {
          bool performCopy = gpudw->compareAndSwapCopyingIntoCPU(
              varName.c_str(), patchID, matlID, levelID);
          if (performCopy) {
            ReductionVariableBase *hostReductionVar =
                dynamic_cast<ReductionVariableBase *>(
                    dependantVar->m_var->typeDescription()->createInstance());
            const bool finalized = dw->isFinalized();
            if (finalized) {
              dw->unfinalize();
            }
            dw->put(*hostReductionVar, dependantVar->m_var, patch->getLevel(),
                    matlID);
            if (finalized) {
              dw->refinalize();
            }
            host_ptr = hostReductionVar->getBasePointer();
            host_bytes = hostReductionVar->getDataSize();

            GPUReductionVariableBase *gpuReductionVar =
                OnDemandDataWarehouse::createGPUReductionVariable(datatype);
            gpudw->get(*gpuReductionVar, varName.c_str(), patchID, matlID,
                       levelID);
            device_ptr = gpuReductionVar->getVoidPointer();
            std::size_t device_bytes = gpuReductionVar->getMemSize();
            delete gpuReductionVar;

            if (host_bytes == device_bytes) {
              stream->memcpy(host_ptr, device_ptr, host_bytes);
              dtask->getVarsBeingCopiedByTask().add(
                patch, matlID, levelID, host_bytes, host_bytes, dependantVar,
                deviceNum, hostReductionVar,
                GpuUtilities::sameDeviceSameMpiRank);
            } else {
              SCI_THROW(InternalError(
                          "InitiateD2H - Reduction variable memory sizes didn't match",
                          __FILE__, __LINE__));
            }
            // delete hostReductionVar;
          }
          break;
        }
        default: {
        }
        }
      }
    }
  }
}

void SYCLScheduler::createTaskGpuDWs(DetailedTask *dtask) {
  // Create GPU datawarehouses for this specific task only. They will get
  // copied into the GPU. This is sizing these datawarehouses dynamically and
  // doing it all in only one alloc per datawarehouse. See the bottom of the
  // GPUDataWarehouse.h for more information.

  std::set<int> deviceNums = dtask->getDeviceNums();
  for (const auto deviceNums_it : deviceNums) {
    const int currentDevice = *deviceNums_it;

    int numItemsInDW =
        dtask->getTaskVars().getTotalVars(currentDevice, Task::OldDW) +
        dtask->getGhostVars().getNumGhostCellCopies(currentDevice, Task::OldDW);
    if (numItemsInDW > 0) {
      std::size_t objectSizeInBytes =
          sizeof(GPUDataWarehouse) -
          sizeof(GPUDataWarehouse::dataItem) * MAX_VARDB_ITEMS +
          sizeof(GPUDataWarehouse::dataItem) * numItemsInDW;

      GPUDataWarehouse *old_taskGpuDW = (GPUDataWarehouse *)malloc(objectSizeInBytes);
      std::ostringstream out;
      out << "Old task GPU DW"
          << " MPIRank: " << Uintah::Parallel::getMPIRank()
          << " Task: " << dtask->getTask()->getName();
      old_taskGpuDW->init(currentDevice, out.str());
      old_taskGpuDW->init_device(objectSizeInBytes, numItemsInDW);
      dtask->setTaskGpuDataWarehouse(currentDevice, Task::OldDW, old_taskGpuDW);
    }

    numItemsInDW =
        dtask->getTaskVars().getTotalVars(currentDevice, Task::NewDW) +
        dtask->getGhostVars().getNumGhostCellCopies(currentDevice, Task::NewDW);
    if (numItemsInDW > 0) {
      std::size_t objectSizeInBytes =
          sizeof(GPUDataWarehouse) -
          sizeof(GPUDataWarehouse::dataItem) * MAX_VARDB_ITEMS +
          sizeof(GPUDataWarehouse::dataItem) * numItemsInDW;

      GPUDataWarehouse *new_taskGpuDW = (GPUDataWarehouse *)malloc(objectSizeInBytes);
      std::ostringstream out;
      out << "New task GPU DW"
          << " MPIRank: " << Uintah::Parallel::getMPIRank()
          << " Thread:" << Impl::t_tid << " Task: " << dtask->getName();
      new_taskGpuDW->init(currentDevice, out.str());
      new_taskGpuDW->init_device(objectSizeInBytes, numItemsInDW);
      dtask->setTaskGpuDataWarehouse(currentDevice, Task::NewDW, new_taskGpuDW);
    }
  }
}

void SYCLScheduler::assignDevicesAndStreams(DetailedTask *dtask) {
  // Figure out which device this patch was assigned to.
  // If a task has multiple patches, then assign all. Most tasks should
  // only end up on one device.  Only tasks like data archiver's output
  // variables work on multiple patches which can be on multiple devices.
  for (int patchID = 0; patchID < dtask->getPatches()->size(); patchID++) {
    const Patch *patch = dtask->getPatches()->get(patchID);
    // Note: on a given device, there is just 1 GPU stream per device per task
    int gpuID = GpuUtilities::getGpuIndexForPatch(patch);
    dtask->setGpuStreamForThisTask(gpuID, GPUStreamPool<>::getInstance().getGpuStreamFromPool(gpuID));
  }
}

void SYCLScheduler::assignDevicesAndStreamsFromGhostVars(DetailedTask *dtask) {
  //  Go through the ghostVars collection and look at the patch where all ghost
  //  cells are going.
  std::set<unsigned int> &destinationDevices = dtask->getGhostVars().getDestinationDevices();
  for (auto iter : destinationDevices) {
    dtask->setGpuStreamForThisTask(*iter, GPUStreamPool<>::getInstance().getGpuStreamFromPool(*iter));
  }
}

void SYCLScheduler::findIntAndExtGpuDependencies(DetailedTask *dtask, int iteration, int t_id) {
  dtask->clearPreparationCollections();

  // Prepare internal dependencies.  Only makes sense if we have multiple GPUs
  // that we are using.
  if (Uintah::Parallel::usingDevice()) {

    // Prepare external dependencies.  The only thing that needs to be
    // prepared is getting ghost cell data from a GPU into a flat
    // array and copied to host memory so that the MPI engine can
    // treat it normally.  That means this handles GPU->other node GPU
    // and GPU->other node CPU.

    for (DependencyBatch *batch = dtask->getComputes(); batch != 0;
         batch = batch->m_comp_next) {
      for (DetailedDep *req = batch->m_head; req != 0; req = req->m_next) {
        if ((req->m_comm_condition == DetailedDep::FirstIteration &&
             iteration > 0) ||
            (req->m_comm_condition == DetailedDep::SubsequentIterations &&
             iteration == 0) ||
            (m_no_copy_data_vars.count(req->m_req->m_var->getName()) > 0)) {
          // See comment in DetailedDep about CommCondition
          continue;
        }

        // if we send/recv to an output task, don't send/recv if not an output
        // timestep

        // ARS NOTE: Outputing and Checkpointing may be done out of
        // snyc now. I.e. turned on just before it happens rather than
        // turned on before the task graph execution.  As such, one
        // should also be checking:

        // m_application->activeReductionVariable( "outputInterval" );
        // m_application->activeReductionVariable( "checkpointInterval" );

        // However, if active the code below would be called regardless
        // if an output or checkpoint time step or not. Not sure that is
        // desired but not sure of the effect of not calling it and doing
        // an out of sync output or checkpoint.

        if (req->m_to_tasks.front()->getTask()->getType() == Task::Output &&
            !m_output->isOutputTimeStep() &&
            !m_output->isCheckpointTimeStep()) {
          continue;
        }
        OnDemandDataWarehouse *dw =
            m_dws[req->m_req->mapDataWarehouse()].get_rep();

        const VarLabel *posLabel;
        OnDemandDataWarehouse *posDW;

        // the load balancer is used to determine where data was in
        // the old dw on the prev timestep - pass it in if the
        // particle data is on the old dw

        if (!m_reloc_new_pos_label && m_parent_scheduler) {
          posDW = m_dws[req->m_req->m_task->mapDataWarehouse(Task::ParentOldDW)]
                      .get_rep();
          posLabel = m_parent_scheduler->m_reloc_new_pos_label;
        } else {
          // on an output task (and only on one) we require particle
          // variables from the NewDW
          if (req->m_to_tasks.front()->getTask()->getType() == Task::Output) {
            posDW = m_dws[req->m_req->m_task->mapDataWarehouse(Task::NewDW)]
                        .get_rep();
          } else {
            posDW = m_dws[req->m_req->m_task->mapDataWarehouse(Task::OldDW)]
                        .get_rep();
          }
          posLabel = m_reloc_new_pos_label;
        }
        // Load information which will be used to later invoke a
        // kernel to copy this range out of the GPU.
        prepareGpuDependencies(dtask, batch, posLabel, dw, posDW, req,
                               GpuUtilities::anotherMpiRank);
      }
    } // end for (DependencyBatch * batch = task->getComputes() )
  }
}

//______________________________________________________________________
//
void SYCLScheduler::syncTaskGpuDWs(DetailedTask *dtask) {
  // For each GPU datawarehouse, see if there are ghost cells listed to be
  // copied if so, launch a kernel that copies them.
  std::set<int> deviceNums = dtask->getDeviceNums();
  GPUDataWarehouse *taskgpudw;
  for (const auto deviceNums_it : deviceNums) {
    const int currentDevice = *deviceNums_it;
    taskgpudw = dtask->getTaskGpuDataWarehouse(currentDevice, Task::OldDW);
    if (taskgpudw) {
      taskgpudw->syncto_device(dtask->getGpuStreamForThisTask(currentDevice));
    }
    taskgpudw = dtask->getTaskGpuDataWarehouse(currentDevice, Task::NewDW);
    if (taskgpudw) {
      taskgpudw->syncto_device(dtask->getGpuStreamForThisTask(currentDevice));
    }
  }
}

//______________________________________________________________________
//
void SYCLScheduler::performInternalGhostCellCopies(DetailedTask *dtask) {

  // For each GPU datawarehouse, see if there are ghost cells listed to be
  // copied if so, launch a kernel that copies them.
  std::set<int> deviceNums = dtask->getDeviceNums();
  for (auto deviceNums_it : deviceNums) {
    const int currentDevice = *deviceNums_it;
    if (dtask->getTaskGpuDataWarehouse(currentDevice, Task::OldDW) != nullptr &&
        dtask->getTaskGpuDataWarehouse(currentDevice, Task::OldDW)
            ->ghostCellCopiesNeeded()) {
      dtask->getTaskGpuDataWarehouse(currentDevice, Task::OldDW)
          ->copyGpuGhostCellsToGpuVarsInvoker(
              dtask->getGpuStreamForThisTask(currentDevice));
    }

    if (dtask->getTaskGpuDataWarehouse(currentDevice, Task::NewDW) != nullptr &&
        dtask->getTaskGpuDataWarehouse(currentDevice, Task::NewDW)
            ->ghostCellCopiesNeeded()) {
      dtask->getTaskGpuDataWarehouse(currentDevice, Task::NewDW)
          ->copyGpuGhostCellsToGpuVarsInvoker(
              dtask->getGpuStreamForThisTask(currentDevice));
    }
  }
}

//______________________________________________________________________
//
void SYCLScheduler::copyAllGpuToGpuDependences(DetailedTask *dtask) {

  // Iterate through the ghostVars, find all whose destination is another GPU
  // same MPI-rank. Get the destination device, the size And do a straight GPU to
  // GPU copy.
  const std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo>
      &ghostVarMap = dtask->getGhostVars().getMap();
  for (const auto it : ghostVarMap) {
    if (it->second.m_dest == GpuUtilities::anotherDeviceSameMpiRank) {
      // TODO: Needs a particle section

      IntVector ghostLow = it->first.m_sharedLowCoordinates;
      IntVector ghostHigh = it->first.m_sharedHighCoordinates;
      IntVector ghostSize(ghostHigh.x() - ghostLow.x(),
                          ghostHigh.y() - ghostLow.y(),
                          ghostHigh.z() - ghostLow.z());
      sycl::int3 device_source_offset;
      sycl::int3 device_source_size;

      // get the source variable from the source GPU DW
      void *device_source_ptr;
      std::size_t elementDataSize = it->second.m_xstride;
      std::size_t memSize =
          ghostSize.x() * ghostSize.y() * ghostSize.z() * elementDataSize;
      GPUGridVariableBase *device_source_var =
          OnDemandDataWarehouse::createGPUGridVariable(it->second.m_datatype);
      OnDemandDataWarehouseP dw = m_dws[it->first.m_dataWarehouse];
      GPUDataWarehouse *gpudw = dw->getGPUDW(it->second.m_sourceDeviceNum);
      gpudw->getStagingVar(
          *device_source_var, it->first.m_label.c_str(),
          it->second.m_sourcePatchPointer->getID(), it->first.m_matlIndx,
          it->first.m_levelIndx,
          make_int3(ghostLow.x(), ghostLow.y(), ghostLow.z()),
          make_int3(ghostSize.x(), ghostSize.y(), ghostSize.z()));
      device_source_var->getArray3(device_source_offset, device_source_size,
                                   device_source_ptr);

      // Get the destination variable from the destination GPU DW
      gpudw = dw->getGPUDW(it->second.m_destDeviceNum);
      sycl::int3 device_dest_offset;
      sycl::int3 device_dest_size;
      void *device_dest_ptr;
      GPUGridVariableBase *device_dest_var =
          OnDemandDataWarehouse::createGPUGridVariable(it->second.m_datatype);
      gpudw->getStagingVar(
          *device_dest_var, it->first.m_label.c_str(),
          it->second.m_destPatchPointer->getID(), it->first.m_matlIndx,
          it->first.m_levelIndx,
          make_int3(ghostLow.x(), ghostLow.y(), ghostLow.z()),
          make_int3(ghostSize.x(), ghostSize.y(), ghostSize.z()));
      device_dest_var->getArray3(device_dest_offset, device_dest_size,
                                 device_dest_ptr);

      // We can run peer copies from the source or the device stream.  While
      // running it from the device technically is said to be a bit slower, it's
      // likely just to an extra event being created to manage blocking the
      // destination stream. By putting it on the device we are able to not need
      // a synchronize step after all the copies, because any upcoming API call
      // will use the streams and be naturally queued anyway.  When a copy
      // completes, anything placed in the destination stream can then process.
      //   Note: If we move to UVA, then we could just do a straight memcpy

      gpuStream_t* stream = dtask->getGpuStreamForThisTask(it->second.m_destDeviceNum);
      OnDemandDataWarehouse::uintahSetGpuDevice(it->second.m_destDeviceNum);

      auto gpuP2Pcopy = stream->memcpy(device_dest_ptr, device_source_ptr, memSize); // SYCL P2P memcpy
    }
  }
}

//______________________________________________________________________
//
void SYCLScheduler::copyAllExtGpuDependenciesToHost(DetailedTask *dtask) {

  bool copiesExist = false;

  // If we put it in ghostVars, then we copied it to an array on the GPU (D2D).
  // Go through the ones that indicate they are going to another MPI rank.  Copy
  // them out to the host (D2H).  To make the engine cleaner for now, we'll then
  // do a H2H copy step into the variable. In the future, to be more efficient,
  // we could skip the host to host copy and instead have sendMPI() send the
  // array we get from the device instead.
  // To be even more efficient than that, if everything is pinned, unified
  // addressing set up, and CUDA aware MPI used, then we could pull everything
  // out via MPI that way and avoid the manual D2H copy and the H2H copy.
  const std::map<GpuUtilities::GhostVarsTuple, DeviceGhostCellsInfo>
      &ghostVarMap = dtask->getGhostVars().getMap();
  for (const auto it = ghostVarMap) {
    // TODO: Needs a particle section
    if (it->second.m_dest == GpuUtilities::anotherMpiRank) {
      void *host_ptr = nullptr;   // host base pointer to raw data
      void *device_ptr = nullptr; // device base pointer to raw data
      std::size_t host_bytes = 0;
      IntVector host_low, host_high, host_offset, host_size, host_strides;
      sycl::int3 device_offset;
      sycl::int3 device_size;

      // We created a temporary host variable for this earlier,
      // and the deviceVars collection knows about it.  It's set as a foreign
      // var.
      IntVector ghostLow = it->first.m_sharedLowCoordinates;
      IntVector ghostHigh = it->first.m_sharedHighCoordinates;
      IntVector ghostSize(ghostHigh.x() - ghostLow.x(),
                          ghostHigh.y() - ghostLow.y(),
                          ghostHigh.z() - ghostLow.z());
      DeviceGridVariableInfo item = dtask->getDeviceVars().getStagingItem(
          it->first.m_label, it->second.m_sourcePatchPointer,
          it->first.m_matlIndx, it->first.m_levelIndx, ghostLow, ghostSize,
          (const int)it->first.m_dataWarehouse);
      GridVariableBase *tempGhostVar = (GridVariableBase *)item.m_var;

      tempGhostVar->getSizes(host_low, host_high, host_offset, host_size,
                             host_strides);

      host_ptr = tempGhostVar->getBasePointer();
      host_bytes = tempGhostVar->getDataSize();

      // copy the computes data back to the host

      GPUGridVariableBase *device_var =
          OnDemandDataWarehouse::createGPUGridVariable(it->second.m_datatype);
      OnDemandDataWarehouseP dw = m_dws[it->first.m_dataWarehouse];
      GPUDataWarehouse *gpudw = dw->getGPUDW(it->second.m_sourceDeviceNum);
      gpudw->getStagingVar(
          *device_var, it->first.m_label.c_str(),
          it->second.m_sourcePatchPointer->getID(), it->first.m_matlIndx,
          it->first.m_levelIndx,
          make_int3(ghostLow.x(), ghostLow.y(), ghostLow.z()),
          make_int3(ghostSize.x(), ghostSize.y(), ghostSize.z()));
      device_var->getArray3(device_offset, device_size, device_ptr);

      // if offset and size is equal to CPU DW, directly copy back to CPU var
      // memory;
      if (device_offset.x() == host_low.x() &&
          device_offset.y() == host_low.y() &&
          device_offset.z() == host_low.z() &&
          device_size.x() == host_size.x() &&
          device_size.y() == host_size.y() &&
          device_size.z() == host_size.z()) {

        // Since we know we need a stream, obtain one.
        gpuStream_t *stream =
            dtask->getGpuStreamForThisTask(it->second.m_sourceDeviceNum);
        OnDemandDataWarehouse::uintahSetGpuDevice(it->second.m_sourceDeviceNum);
        // aysnc
        stream->memcpy(host_ptr, device_ptr, host_bytes);

        copiesExist = true;
      } else {
        std::cerr
            << "SYCLScheduler::GpuDependenciesToHost() - Error - The host "
               "and device variable sizes did not match.  Cannot copy D2H."
            << std::endl;
        SCI_THROW(InternalError("Error - The host and device variable sizes "
                                "did not match.  Cannot copy D2H",
                                __FILE__, __LINE__));
      }

      delete device_var;
    }
  }

  if (copiesExist) {

    // Wait until all streams are done
    // Further optimization could be to check each stream one by one and make
    // copies before waiting for other streams to complete.
    // TODO: There's got to be a better way to do this.
    while (!dtask->checkAllGpuStreamsDoneForThisTask()) {
      // TODO - Let's figure this out soon, APH 06/09/16
      // sleep?
      // printf("Sleeping\n");
    }

    for (const auto it : ghostVarMap) {

      if (it->second.m_dest == GpuUtilities::anotherMpiRank) {
        // TODO: Needs a particle section
        IntVector host_low, host_high, host_offset, host_size, host_strides;
        OnDemandDataWarehouseP dw = m_dws[(const int)it->first.m_dataWarehouse];

        // We created a temporary host variable for this earlier,
        // and the deviceVars collection knows about it.
        IntVector ghostLow = it->first.m_sharedLowCoordinates;
        IntVector ghostHigh = it->first.m_sharedHighCoordinates;
        IntVector ghostSize(ghostHigh.x() - ghostLow.x(),
                            ghostHigh.y() - ghostLow.y(),
                            ghostHigh.z() - ghostLow.z());
        DeviceGridVariableInfo item = dtask->getDeviceVars().getStagingItem(
            it->first.m_label, it->second.m_sourcePatchPointer,
            it->first.m_matlIndx, it->first.m_levelIndx, ghostLow, ghostSize,
            (const int)it->first.m_dataWarehouse);
        GridVariableBase *tempGhostVar = (GridVariableBase *)item.m_var;

        // Also get the existing host copy
        GridVariableBase *gridVar = dynamic_cast<GridVariableBase *>(
            it->second.m_label->typeDescription()->createInstance());

        // Get the coordinate low/high of the host copy.
        const Patch *patch = it->second.m_sourcePatchPointer;
        TypeDescription::Type type =
            it->second.m_label->typeDescription()->getType();
        IntVector lowIndex, highIndex;
        bool uses_SHRT_MAX = (item.m_dep->m_num_ghost_cells == SHRT_MAX);
        if (uses_SHRT_MAX) {
          const Level *level = patch->getLevel();
          level->computeVariableExtents(type, lowIndex, highIndex);
        } else {
          Patch::VariableBasis basis = Patch::translateTypeToBasis(
              it->second.m_label->typeDescription()->getType(), false);
          patch->computeVariableExtents(
              basis, item.m_dep->m_var->getBoundaryLayer(), item.m_dep->m_gtype,
              item.m_dep->m_num_ghost_cells, lowIndex, highIndex);
        }

        // If it doesn't exist yet on the host, create it.  If it does exist on
        // the host, then if we got here that meant the host data was invalid
        // and the device data was valid, so nuke the old contents and create a
        // new one.  (Should we just get a mutable var instead as it should be
        // the size we already need?)  This process is admittedly a bit hacky,
        // as now the var will be both partially valid and invalid.  The ghost
        // cell region is now valid on the host, while the rest of the host var
        // would be invalid. Since we are writing to an old data warehouse (from
        // device to host), we need to temporarily unfinalize it.
        const bool finalized = dw->isFinalized();
        if (finalized) {
          dw->unfinalize();
        }

        if (!dw->exists(item.m_dep->m_var, it->first.m_matlIndx,
                        it->second.m_sourcePatchPointer)) {
          dw->allocateAndPut(*gridVar, item.m_dep->m_var, it->first.m_matlIndx,
                             it->second.m_sourcePatchPointer,
                             item.m_dep->m_gtype,
                             item.m_dep->m_num_ghost_cells);
        } else {
          // Get a const variable in a non-constant way.
          // This assumes the variable has already been resized properly, which
          // is why ghost cells are set to zero.
          // TODO: Check sizes anyway just to be safe.
          dw->getModifiable(*gridVar, item.m_dep->m_var, it->first.m_matlIndx,
                            it->second.m_sourcePatchPointer, Ghost::None, 0);
        }
        // Do a host-to-host copy to bring the device data now on the host into
        // the host-side variable so that sendMPI can easily find the data as if
        // no GPU were involved at all.
        gridVar->copyPatch(tempGhostVar, ghostLow, ghostHigh);
        if (finalized) {
          dw->refinalize();
        }

        // let go of our reference counters.
        delete gridVar;
        delete tempGhostVar;
      }
    }
  }
}

//______________________________________________________________________
//  generate string   <MPI_rank>.<Thread_ID>
std::string SYCLScheduler::myRankThread() {
  std::ostringstream out;
  out << Uintah::Parallel::getMPIRank() << "." << Impl::t_tid;
  return out.str();
}

void SYCLScheduler::init_threads(SYCLScheduler *sched, int num_threads) {
  Impl::init_threads(sched, num_threads);
}

SYCLSchedulerWorker::SYCLSchedulerWorker(SYCLScheduler *scheduler, int tid,
                                         int affinity)
  : m_scheduler{scheduler}, m_rank{scheduler->d_myworld->myRank()} {}

void SYCLSchedulerWorker::run() {
  while (Impl::g_run_tasks.load(std::memory_order_relaxed) == 1) {
    try {
      m_scheduler->runTasks(Impl::t_tid);
    } catch (Exception &e) {
      std::exit(-1);
    }
  }
}
