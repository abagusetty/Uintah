#include <CCA/Components/Arches/CoalModels/SimpleHeatTransfer.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/TransportEqns/DQMOMEqn.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Parallel/Parallel.h>
//#include <iomanip>

//===========================================================================

using namespace std;
using namespace Uintah; 

//---------------------------------------------------------------------------
// Builder:
SimpleHeatTransferBuilder::SimpleHeatTransferBuilder( const std::string         & modelName,
                                                      const vector<std::string> & reqICLabelNames,
                                                      const vector<std::string> & reqScalarLabelNames,
                                                      const ArchesLabel         * fieldLabels,
                                                      SimulationStateP          & sharedState,
                                                      int qn ) :
  ModelBuilder( modelName, reqICLabelNames, reqScalarLabelNames, fieldLabels, sharedState, qn )
{
}

SimpleHeatTransferBuilder::~SimpleHeatTransferBuilder(){}

ModelBase* SimpleHeatTransferBuilder::build() {
  return scinew SimpleHeatTransfer( d_modelName, d_sharedState, d_fieldLabels, d_icLabels, d_scalarLabels, d_quadNode );
}
// End Builder
//---------------------------------------------------------------------------

SimpleHeatTransfer::SimpleHeatTransfer( std::string modelName, 
                                        SimulationStateP& sharedState,
                                        const ArchesLabel* fieldLabels,
                                        vector<std::string> icLabelNames, 
                                        vector<std::string> scalarLabelNames,
                                        int qn ) 
: HeatTransfer(modelName, sharedState, fieldLabels, icLabelNames, scalarLabelNames, qn)
{
  // Model/gas labels created in parent class

  // Set constants
  Pr = 0.7;
  blow = 1.0;
  sigma = 5.67e-8;   // [=] J/s/m^2/K^4 : Stefan-Boltzmann constant (from white book)
  //rkg = 0.03;        // [=] J/s/m/K : thermal conductivity of gas

  pi = 3.14159265358979; 
}

SimpleHeatTransfer::~SimpleHeatTransfer()
{
  VarLabel::destroy(d_abskp);
}

//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void 
SimpleHeatTransfer::problemSetup(const ProblemSpecP& params)
{
  HeatTransfer::problemSetup( params );

  ProblemSpecP db = params; 
  
  // check for viscosity
  const ProblemSpecP params_root = db->getRootNode(); 
  if (params_root->findBlock("PhysicalConstants")) {
    ProblemSpecP db_phys = params_root->findBlock("PhysicalConstants");
    db_phys->require("viscosity", visc);
    if( visc == 0 ) {
      throw InvalidValue("ERROR: SimpleHeatTransfer: problemSetup(): Zero viscosity specified in <PhysicalConstants> section of input file.",__FILE__,__LINE__);
    }
  } else {
    throw InvalidValue("ERROR: SimpleHeatTransfer: problemSetup(): Missing <PhysicalConstants> section in input file, no viscosity value specified.",__FILE__,__LINE__);
  }

  if (params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties")) {
    ProblemSpecP db_coal = params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("Coal_Properties");
    db_coal->require("C", yelem[0]);
    db_coal->require("H", yelem[1]);
    db_coal->require("N", yelem[2]);
    db_coal->require("O", yelem[3]);
    db_coal->require("S", yelem[4]);
    db_coal->require("initial_ash_mass", ash_mass_init);

    // normalize amounts
    double ysum = yelem[0] + yelem[1] + yelem[2] + yelem[3] + yelem[4];
    yelem[0] = yelem[0]/ysum;
    yelem[1] = yelem[1]/ysum;
    yelem[2] = yelem[2]/ysum;
    yelem[3] = yelem[3]/ysum;
    yelem[4] = yelem[4]/ysum;
  } else {
    throw InvalidValue("ERROR: SimpleHeatTransfer: problemSetup(): Missing <Coal_Properties> section in input file. Please specify the elemental composition of the coal and the initial ash mass.",__FILE__,__LINE__);
  }

  // Check for radiation 
  b_radiation = false;
  if (params_root->findBlock("CFD")->findBlock("ARCHES")->findBlock("ExplicitSolver")->findBlock("EnthalpySolver")->findBlock("DORadiationModel"))
    b_radiation = true; // if gas phase radiation is turned on.  
  //user can specifically turn off radiation heat transfer
  if (db->findBlock("noRadiation"))
    b_radiation = false;


  // Assume no ash (for now)
  //d_ash = false;

  string label_name;
  string role_name;
  string temp_label_name;

  string temp_ic_name;
  string temp_ic_name_full;

  std::stringstream out;
  out << d_quadNode; 
  string node = out.str();

  // -----------------------------------------------------------------
  // Look for required internal coordinates
  ProblemSpecP db_icvars = params->findBlock("ICVars");
  if (db_icvars) {
    for (ProblemSpecP variable = db_icvars->findBlock("variable"); variable != 0; variable = variable->findNextBlock("variable") ) {
    
      variable->getAttribute("label",label_name);
      variable->getAttribute("role",role_name);

      temp_label_name = label_name;
      temp_label_name += node;

      // user specifies "role" of each internal coordinate
      if( role_name == "particle_length" 
               //|| role_name == "char_mass"
               //|| role_name == "ash_mass"
               //|| role_name == "moisture_mass"
               || role_name == "particle_temperature"
               || role_name == "raw_coal_mass" ) {
        LabelToRoleMap[temp_label_name] = role_name;
      } else {
        std::string errmsg = "ERROR: SimpleHeatTransfer: problemSetup(): Invalid variable role for Simple Heat Transfer model!";
        throw InvalidValue(errmsg,__FILE__,__LINE__);
      }

      // set model clipping
      db->getWithDefault( "low_clip",  d_lowModelClip,  1.0e-6 );
      db->getWithDefault( "high_clip", d_highModelClip, 999999 );
    }
  }

  // fix the d_icLabels to point to the correct quadrature node (since there is 1 model per quad node)
  for ( vector<std::string>::iterator iString = d_icLabels.begin(); 
        iString != d_icLabels.end(); ++iString) {
    
    temp_ic_name      = (*iString);
    temp_ic_name_full = temp_ic_name;
    temp_ic_name_full += node;

    std::replace( d_icLabels.begin(), d_icLabels.end(), temp_ic_name, temp_ic_name_full);
  }

  // -----------------------------------------------------------------
  // Look for required scalars
  ProblemSpecP db_scalarvars = params->findBlock("scalarVars");
  if (db_scalarvars) {
    for( ProblemSpecP variable = db_scalarvars->findBlock("variable");
         variable != 0; variable = variable->findNextBlock("variable") ) {

      variable->getAttribute("label", label_name);
      variable->getAttribute("role",  role_name);

      temp_label_name = label_name;
      temp_label_name += node;

      // user specifies "role" of each scalar
      if( role_name == "particle_temperature" 
          || role_name == "gas_temperature" ) {
        LabelToRoleMap[temp_label_name] = role_name;
      } else {
        std::string errmsg;
        errmsg = "Invalid scalar variable role for Simple Heat Transfer model: must be \"particle_temperature\" or \"gas_temperature\", you specified \"" + role_name + "\".";
        throw InvalidValue(errmsg,__FILE__,__LINE__);
      }
    }
  }


  ///////////////////////////////////////////


  DQMOMEqnFactory& dqmom_eqn_factory = DQMOMEqnFactory::self();
  EqnFactory& eqn_factory = EqnFactory::self();

  // assign labels for each required internal coordinate
  for( map<string,string>::iterator iter = LabelToRoleMap.begin();
       iter != LabelToRoleMap.end(); ++iter ) {

    if ( iter->second == "particle_temperature") {
      if( dqmom_eqn_factory.find_scalar_eqn(iter->first) ) {
        EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(iter->first);
        DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
        d_particle_temperature_label = current_eqn.getTransportEqnLabel();
        d_pt_scaling_constant = current_eqn.getScalingConstant();

      } else if( eqn_factory.find_scalar_eqn(iter->first) ) {
        EqnBase& t_current_eqn = eqn_factory.retrieve_scalar_eqn(iter->first);
        d_particle_temperature_label = t_current_eqn.getTransportEqnLabel();
        d_pt_scaling_constant = t_current_eqn.getScalingConstant();
      
      } else {
        std::string errmsg = "ARCHES: SimpleHeatTransfer: Invalid variable given for <variable> tag in SimpleHeatTransfer model.";
        errmsg += "\nCould not find given particle temperature variable \"";
        errmsg += iter->second;
        errmsg += "\" in DQMOMEqnFactory.";
        throw ProblemSetupException(errmsg,__FILE__,__LINE__);
      }

    } else if( iter->second == "particle_length" ) {
      if (dqmom_eqn_factory.find_scalar_eqn(iter->first) ) {
        EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(iter->first);
        DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
        d_particle_length_label = current_eqn.getTransportEqnLabel();
        d_pl_scaling_constant = current_eqn.getScalingConstant();
      } else {
        std::string errmsg = "ARCHES: SimpleHeatTransfer: Invalid variable given for <variable> tag in SimpleHeatTransfer model.";
        errmsg += "\nCould not find given particle length variable \"";
        errmsg += iter->second;
        errmsg += "\" in DQMOMEqnFactory.";
        throw ProblemSetupException(errmsg,__FILE__,__LINE__);
      }
    
    } else if ( iter->second == "raw_coal_mass") {
      if (dqmom_eqn_factory.find_scalar_eqn(iter->first) ) {
        EqnBase& t_current_eqn = dqmom_eqn_factory.retrieve_scalar_eqn(iter->first);
        DQMOMEqn& current_eqn = dynamic_cast<DQMOMEqn&>(t_current_eqn);
        d_raw_coal_mass_label = current_eqn.getTransportEqnLabel();
        d_rc_scaling_constant = current_eqn.getScalingConstant();
      } else {
        std::string errmsg = "ARCHES: SimpleHeatTransfer: Invalid variable given for <variable> tag in SimpleHeatTransfer model.";
        errmsg += "\nCould not find given coal mass  variable \"";
        errmsg += iter->second;
        errmsg += "\" in DQMOMEqnFactory.";
        throw ProblemSetupException(errmsg,__FILE__,__LINE__);
      }
    } else {
      // can't find this required variable in the labels-to-roles map!
      std::string errmsg = "ARCHES: SimpleHeatTransfer: You specified that the variable \"" + iter->first + 
                           "\" was required, but you did not specify a role for it!\n";
      throw ProblemSetupException( errmsg, __FILE__, __LINE__);
    }
  }


  // thermal conductivity of particles
  std::string abskpName = "abskp_qn";
  abskpName += node; 
  d_abskp = VarLabel::create(abskpName, CCVariable<double>::getTypeDescription());

}

//---------------------------------------------------------------------------
// Method: Schedule the initialization of special variables unique to model
//---------------------------------------------------------------------------
void 
SimpleHeatTransfer::sched_initVars( const LevelP& level, SchedulerP& sched )
{
  std::string taskname = "SimpleHeatTransfer::initVars";
  Task* tsk = scinew Task(taskname, this, &SimpleHeatTransfer::initVars);

  tsk->computes( d_modelLabel );
  tsk->computes( d_gasLabel   );
  tsk->computes(d_abskp);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 
}

//-------------------------------------------------------------------------
// Method: Initialize special variables unique to the model
//-------------------------------------------------------------------------
void
SimpleHeatTransfer::initVars( const ProcessorGroup * pc, 
                              const PatchSubset    * patches, 
                              const MaterialSubset * matls, 
                              DataWarehouse        * old_dw, 
                              DataWarehouse        * new_dw )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> model_value; 
    new_dw->allocateAndPut( model_value, d_modelLabel, matlIndex, patch ); 
    model_value.initialize(0.0);

    CCVariable<double> gas_value; 
    new_dw->allocateAndPut( gas_value, d_gasLabel, matlIndex, patch ); 
    gas_value.initialize(0.0);

    CCVariable<double> abskp; 
    new_dw->allocateAndPut( abskp, d_abskp, matlIndex, patch ); 
    abskp.initialize(0.0);

  }
}

//---------------------------------------------------------------------------
// Method: Schedule the calculation of the Model 
//---------------------------------------------------------------------------
void 
SimpleHeatTransfer::sched_computeModel( const LevelP& level, SchedulerP& sched, int timeSubStep )
{
  std::string taskname = "SimpleHeatTransfer::computeModel";
  Task* tsk = scinew Task(taskname, this, &SimpleHeatTransfer::computeModel);

  d_timeSubStep = timeSubStep; 

  if (d_timeSubStep == 0 && !d_labelSchedInit) {
    // Every model term needs to set this flag after the varLabel is computed. 
    // transportEqn.cleanUp should reinitialize this flag at the end of the time step. 
    d_labelSchedInit = true;

    tsk->computes(d_modelLabel);
    tsk->computes(d_gasLabel); 
    tsk->computes(d_abskp);
  } else {
    tsk->modifies(d_modelLabel);
    tsk->modifies(d_gasLabel);  
    tsk->modifies(d_abskp);
  }

  CoalModelFactory& coalFactory = CoalModelFactory::self();

  // also require paticle velocity, gas velocity, and density
  if( coalFactory.useParticleVelocityModel() ) {
    tsk->requires( Task::OldDW, coalFactory.getParticleVelocityLabel( d_quadNode), Ghost::None, 0 );
  }
  tsk->requires( Task::OldDW, d_fieldLabels->d_newCCVelocityLabel, Ghost::None, 0);
  tsk->requires(Task::OldDW, d_fieldLabels->d_densityCPLabel, Ghost::None, 0);
  tsk->requires(Task::OldDW, d_fieldLabels->d_cpINLabel, Ghost::None, 0);
 
  if(b_radiation){
    tsk->requires(Task::OldDW, d_fieldLabels->d_radiationSRCINLabel,  Ghost::None, 0);
    tsk->requires(Task::OldDW, d_fieldLabels->d_abskgINLabel,  Ghost::None, 0);   
    tsk->requires(Task::OldDW, d_fieldLabels->d_radiationVolqINLabel, Ghost::None, 0);
  }

  tsk->requires(Task::OldDW, d_particle_temperature_label, Ghost::None, 0);
  tsk->requires(Task::OldDW, d_fieldLabels->d_tempINLabel, Ghost::None, 0);

  // require internal coordinates
  tsk->requires(Task::OldDW, d_weight_label, Ghost::None, 0);
  tsk->requires(Task::OldDW, d_particle_length_label, Ghost::None, 0);
  tsk->requires(Task::OldDW, d_raw_coal_mass_label, Ghost::None, 0);

  sched->addTask(tsk, level->eachPatch(), d_sharedState->allArchesMaterials()); 

}

//---------------------------------------------------------------------------
// Method: Actually compute the source term 
//---------------------------------------------------------------------------
void
SimpleHeatTransfer::computeModel( const ProcessorGroup * pc, 
                                  const PatchSubset    * patches, 
                                  const MaterialSubset * matls, 
                                  DataWarehouse        * old_dw, 
                                  DataWarehouse        * new_dw )
{
  for( int p=0; p < patches->size(); p++ ) {  // Patch loop

    //Ghost::GhostType  gaf = Ghost::AroundFaces;
    //Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gn  = Ghost::None;

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    CCVariable<double> heat_rate;
    if ( new_dw->exists( d_modelLabel, matlIndex, patch) ) {
      new_dw->getModifiable( heat_rate, d_modelLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( heat_rate, d_modelLabel, matlIndex, patch );
      heat_rate.initialize(0.0);
    }
    
    CCVariable<double> gas_heat_rate; 
    if( new_dw->exists( d_gasLabel, matlIndex, patch ) ) {
      new_dw->getModifiable( gas_heat_rate, d_gasLabel, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( gas_heat_rate, d_gasLabel, matlIndex, patch );
      gas_heat_rate.initialize(0.0);
    }
    
    CCVariable<double> abskp; 
    if( new_dw->exists( d_abskp, matlIndex, patch) ) {
      new_dw->getModifiable( abskp, d_abskp, matlIndex, patch ); 
    } else {
      new_dw->allocateAndPut( abskp, d_abskp, matlIndex, patch );
      abskp.initialize(0.0);
    }
    
    
    CoalModelFactory& coalFactory = CoalModelFactory::self();

    // get particle velocity used to calculate Reynolds number
    constCCVariable<Vector> partVel;  
    if( coalFactory.useParticleVelocityModel() ) {
      old_dw->get( partVel, coalFactory.getParticleVelocityLabel( d_quadNode ), matlIndex, patch, gn, 0 );
    } else {
      old_dw->get( partVel, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0 );
    }
    
    // gas velocity used to calculate Reynolds number
    constCCVariable<Vector> gasVel; 
    old_dw->get( gasVel, d_fieldLabels->d_newCCVelocityLabel, matlIndex, patch, gn, 0 ); 
    
    constCCVariable<double> den;
    old_dw->get(den, d_fieldLabels->d_densityCPLabel, matlIndex, patch, gn, 0 ); 
    constCCVariable<double> cpg;
    old_dw->get(cpg, d_fieldLabels->d_cpINLabel, matlIndex, patch, gn, 0 );

    constCCVariable<double> radiationSRCIN;
    constCCVariable<double> abskgIN;
    constCCVariable<double> radiationVolqIN;

    if(b_radiation){
      old_dw->get(radiationSRCIN, d_fieldLabels->d_radiationSRCINLabel, matlIndex, patch, gn, 0);
      old_dw->get(abskgIN, d_fieldLabels->d_abskgINLabel, matlIndex, patch, gn, 0);
      old_dw->get(radiationVolqIN, d_fieldLabels->d_radiationVolqINLabel, matlIndex, patch, gn, 0);
    }

    constCCVariable<double> temperature;
    constCCVariable<double> w_particle_temperature;
    constCCVariable<double> w_particle_length;
    constCCVariable<double> w_raw_coal_mass;
    constCCVariable<double> weight;

    old_dw->get( temperature, d_fieldLabels->d_tempINLabel, matlIndex, patch, gn, 0 );
    old_dw->get( w_particle_temperature, d_particle_temperature_label, matlIndex, patch, gn, 0 );
    old_dw->get( w_particle_length, d_particle_length_label, matlIndex, patch, gn, 0 );
    old_dw->get( w_raw_coal_mass, d_raw_coal_mass_label, matlIndex, patch, gn, 0 );
    old_dw->get( weight, d_weight_label, matlIndex, patch, gn, 0 );

#if !defined(VERIFY_SIMPLEHEATTRANSFER_MODEL)

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter; 
      IntVector cxp = *iter + IntVector(1,0,0);
      IntVector cxm = *iter - IntVector(1,0,0);
      IntVector cyp = *iter + IntVector(0,1,0);
      IntVector cym = *iter - IntVector(0,1,0);
      IntVector czp = *iter + IntVector(0,0,1);
      IntVector czm = *iter - IntVector(0,0,1);
      // define variables specific for non-verification runs:

      // velocities
      Vector gas_velocity = gasVel[c];
      Vector particle_velocity = partVel[c];

      // weight - check if small
      bool weight_is_small = (weight[c] < d_w_small);

      double scaled_weight;
      double unscaled_weight;
      if (weight_is_small) {
        scaled_weight = 0.0;
        unscaled_weight = 0.0;
      } else {
        scaled_weight = weight[c];
        unscaled_weight = weight[c]*d_w_scaling_constant;
      }

      // temperature - particle
      double scaled_particle_temperature;
      double unscaled_particle_temperature;
      if (weight_is_small) {
        scaled_particle_temperature = 0.0;
        unscaled_particle_temperature = 0.0;
      } else {
        scaled_particle_temperature = (w_particle_temperature[c])/scaled_weight;
        unscaled_particle_temperature = (w_particle_temperature[c]*d_pt_scaling_constant)/scaled_weight;
      }

      // temperature - gas
      double gas_temperature = temperature[c];

      // paticle length
      double scaled_length;
      double unscaled_length;
      if (weight_is_small) {
        scaled_length = 0.0;
        unscaled_length = 0.0;
      } else {
        scaled_length = w_particle_length[c]/scaled_weight;
        unscaled_length = (w_particle_length[c]*d_pl_scaling_constant)/scaled_weight;
      }

      // particle raw coal mass
      double scaled_raw_coal_mass;
      double unscaled_raw_coal_mass;
      if (weight_is_small) {
        scaled_raw_coal_mass = 0.0;
        unscaled_raw_coal_mass = 0.0;
      } else {
        scaled_raw_coal_mass = w_raw_coal_mass[c]/scaled_weight;
        unscaled_raw_coal_mass = (w_raw_coal_mass[c]*d_rc_scaling_constant)/scaled_weight;
      }

      double unscaled_ash_mass = ash_mass_init[d_quadNode];
      double density = den[c];
      // viscosity should be grabbed from data warehouse... right now it's constant

      double FSum = 0.0;

      double heat_rate_;
      double gas_heat_rate_;
      double abskp_;

#else
      Vector gas_velocity       = Vector(3.0);
      Vector particle_velocity  = Vector(1.0);

      bool weight_is_small = false;

      double unscaled_weight = 1.0e6;
      double scaled_weight = unscaled_weight;

      double unscaled_particle_temperature = 2000.0;
      double scaled_particle_temperature = unscaled_particle_temperature;
      double d_pt_scaling_constant = 1.0;

      double gas_temperature = 2050;

      double unscaled_length = 1.0e-5;
      double scaled_length = unscaled_length;

      double unscaled_raw_coal_mass = 1.0e-8;
      double scaled_raw_coal_mass = unscaled_raw_coal_mass;

      double unscaled_ash_mass = 1.0e-9;
      double scaled_ash_mass = unscaled_ash_mass;

      double density = 1;
      visc = 1.0e-5;

      // redefine composition array
      yelem[0] = 0.75; // C
      yelem[1] = 0.05; // H
      yelem[2] = 0.00; // N
      yelem[3] = 0.20; // O
      yelem[4] = 0.00; // S

      double FSum = 0.0;

      double heat_rate_;
      double gas_heat_rate_;
      double abskp_;

#endif

      // intermediate calculation values
      double Re;
      double Nu;
      double Cpc;
      double Cpa; 
      double mp_Cp;
      double rkg;
      double Q_convection;
      double Q_radiation;

      if (weight_is_small) {
        heat_rate_ = 0.0;
        gas_heat_rate_ = 0.0;
        abskp_ = 0.0;
      } else {

        // ---------------------------------------------
        // Convection part: 

        // Reynolds number
        Re = abs(gas_velocity.length() - particle_velocity.length())*unscaled_length*density/visc;

        // Nusselt number
        Nu = 2.0 + 0.65*pow(Re,0.50)*pow(Pr,(1.0/3.0));

        // Heat capacity of raw coal
        Cpc = heatcp( unscaled_particle_temperature );

        // Heat capacity of ash
        Cpa = heatap( unscaled_particle_temperature );

        // Heat capacity
        mp_Cp = (Cpc*unscaled_raw_coal_mass + Cpa*unscaled_ash_mass);

        // Gas thermal conductivity
        rkg = props(gas_temperature, unscaled_particle_temperature); // [=] J/s/m/K

        // Q_convection (see Section 5.4 of LES_Coal document)
        Q_convection = Nu*pi*blow*rkg*unscaled_length*(gas_temperature - unscaled_particle_temperature);


        // ---------------------------------------------
        // Radiation part: 

        Q_radiation = 0.0;
        if (b_radiation) {
          double Qabs = 0.8;
	        double Apsc = (pi/4)*Qabs*pow(unscaled_length,2);
	        double Eb = 4*sigma*pow(unscaled_particle_temperature,4);
          FSum = radiationVolqIN[c];    
	        Q_radiation = Apsc*(FSum - Eb);
	        abskp_ = pi/4*Qabs*unscaled_weight*pow(unscaled_length,2); 
        } else {
          abskp_ = 0.0;
        }
      
        heat_rate_ = (Q_convection + Q_radiation)/(mp_Cp*d_pt_scaling_constant);
        gas_heat_rate_ = 0.0;
 
      }


#if defined(VERIFY_SIMPLEHEATTRANSFER_MODEL)
      proc0cout << "****************************************************************" << endl;
      proc0cout << "Verification error, Simple Heat Transfer model: " << endl;
      proc0cout << endl;

      double error; 
      double verification_value;
      
      verification_value = 2979.4;
      error = ( (verification_value)-(Cpc) )/(verification_value);
      if( fabs(error) < 0.01 ) {
        proc0cout << "Verification for raw coal heat capacity successful:" << endl;
        proc0cout << "    Percent error = " << setw(4) << fabs(error)*100 << " \%, which is less than 1 percent." << endl;
      } else {
        proc0cout << "WARNING: VERIFICATION FOR RAW COAL HEAT CAPACITY FAILED!!! " << endl;
        proc0cout << "    Verification value  = " << verification_value << endl;
        proc0cout << "    Calculated value    = " << Cpc << endl;
        proc0cout << "    Percent error = " << setw(4) << setprecision(4) << fabs(error)*100 << " \%, which is greater than 1 percent." << endl;
      }

      proc0cout << endl;

      verification_value = 1765.00;
      error = ( (verification_value)-(Cpa) )/(verification_value);
      if( fabs(error) < 0.01 ) {
        proc0cout << "Verification for ash heat capacity successful:" << endl;
        proc0cout << "    Percent error = " << setw(4) << setprecision(4) << fabs(error)*100 << " \%, which is less than 1 percent." << endl;
      } else {
        proc0cout << "WARNING: VERIFICATION FOR ASH HEAT CAPACITY FAILED!!! " << endl;
        proc0cout << "    Verification value  = " << verification_value << endl;
        proc0cout << "    Calculated value    = " << Cpa << endl;
        proc0cout << "    Percent error = " << setw(4) << setprecision(4) << fabs(error)*100 << " \%, which is greater than 1 percent." << endl;
      }

      proc0cout << endl;
      
      verification_value = 4.6985e-4;
      error = ( (verification_value)-(Q_convection) )/(verification_value);
      if( fabs(error) < 0.01 ) {
        proc0cout << "Verification for convection heat transfer term successful:" << endl;
        proc0cout << "    Percent error = " << setw(4) << setprecision(4) << fabs(error)*100 << " \%, which is less than 1 percent." << endl;
      } else {
        proc0cout << "WARNING: VERIFICATION FOR CONVECTION HEAT TRANSFER TERM FAILED!!! " << endl;
        proc0cout << "    Verification value  = " << verification_value << endl;
        proc0cout << "    Calculated value    = " << Q_convection << endl;
        proc0cout << "    Percent error = " << setw(4) << setprecision(4) << fabs(error)*100 << " \%, which is greater than 1 percent." << endl;
      }

      proc0cout << endl;

      verification_value = 0.097312;
      error = ( (verification_value)-(rkg) )/(verification_value);
      if( fabs(error) < 0.01 ) {
        proc0cout << "Verification for gas thermal conductivity successful:" << endl;
        proc0cout << "    Percent error = " << setw(5) << setprecision(4) << fabs(error)*100 << " \%, which is less than 1 percent." << endl;
      } else {
        proc0cout << "WARNING: VERIFICATION FOR GAS THERMAL CONDUCTIVITY FAILED!!! " << endl;
        proc0cout << "    Verification value  = " << verification_value << endl;
        proc0cout << "    Calculated value    = " << rkg << endl;
        proc0cout << "    Percent error = " << setw(5) << setprecision(4) << fabs(error)*100 << " \%, which is greater than 1 percent." << endl;
      }

      proc0cout << endl;
      
      verification_value = 14.888;
      error = ( (verification_value)-(heat_rate_) )/(verification_value);
      if( fabs(error) < 0.01 ) {
        proc0cout << "Verification for particle heating rate term successful:" << endl;
        proc0cout << "    Percent error = " << setw(5) << setprecision(4) << fabs(error)*100 << " \%, which is less than 1 percent." << endl;
      } else {
        proc0cout << "WARNING: VERIFICATION FOR PARTICLE HEATING RATE TERM FAILED!!! " << endl;
        proc0cout << "    Verification value  = " << verification_value << endl;
        proc0cout << "    Calculated value    = " << heat_rate_ << endl;
        proc0cout << "    Percent error = " << setw(5) << setprecision(4) << fabs(error)*100 << " \%, which is greater than 1 percent." << endl;
      }

      proc0cout << endl;
      proc0cout << "****************************************************************" << endl;

      proc0cout << endl << endl;

#else
      heat_rate[c] = heat_rate_;
      gas_heat_rate[c] = gas_heat_rate_;
      abskp[c] = abskp_;
 
    }//end cell loop
#endif

  }//end patch loop
}



// ********************************************************
// Private methods:

double
SimpleHeatTransfer::g1( double z){
  double dum1 = (exp(z)-1)/z;
  double dum2 = pow(dum1,2);
  double sol = exp(z)/dum2;
  return sol;
}

double
SimpleHeatTransfer::heatcp(double Tp){
  if (Tp < 273) {
    // correlation is not valid
    return 0.0;
  } else {
    double MW [5] = { 12., 1., 14., 16., 32.}; // Atomic weight of elements (C,H,N,O,S)
    double Rgas = 8314.3; // J/kg/K

    double MW_avg = 0.0; // Mean atomic weight of coal
    for(int i=0;i<5;i++){
      MW_avg += yelem[i]/MW[i];
    }
    MW_avg = 1/MW_avg;

    double z1 = 380.0/Tp;
    double z2 = 1800.0/Tp;
    double cp = (Rgas/MW_avg)*(g1(z1)+2.0*g1(z2));
    return cp; // J/kg/K
  }
}


double
SimpleHeatTransfer::heatap(double Tp){
  // c.f. PCGC2
  double cpa = 593.0 + 0.586*Tp;
  return cpa;  // J/kg/K
}


double
SimpleHeatTransfer::props(double Tg, double Tp){

  double tg0[10] = {300.,  400.,   500.,   600.,  700.,  800.,  900.,  1000., 1100., 1200. };
  double kg0[10] = {.0262, .03335, .03984, .0458, .0512, .0561, .0607, .0648, .0685, .07184};
  double T = (Tp+Tg)/2; // Film temperature

//   CALCULATE UG AND KG FROM INTERPOLATION OF TABLE VALUES FROM HOLMAN
//   FIND INTERVAL WHERE TEMPERATURE LIES. 

  double kg = 0.0;

  if( T > 1200.0 ) {
    kg = kg0[9] * pow( T/tg0[9], 0.58);

  } else if ( T < 300 ) {
    kg = kg0[0];
  
  } else {
    int J = -1;
    for ( int I=0; I < 9; I++ ) {
      if ( T > tg0[I] ) {
        J = J + 1;
      }
    }
    double FAC = ( tg0[J] - T ) / ( tg0[J] - tg0[J+1] );
    kg = ( -FAC*( kg0[J] - kg0[J+1] ) + kg0[J] );
  }

  return kg; // I believe this is in J/s/m/K, but not sure
}

