%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Material Selection Optimization Model with Further Relaxed Constraints
% and Enhanced Diversity (with NamedConstraints Mapping)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Enumerations for categorical properties.
enum CategoryT = { Polymer, Metal, Composite, Superalloy, Advanced, Other, Ceramic, Coating, Carbon };

% Enumeration for base materials.
enum BaseMaterialEnum = {
  AluminumAlloy, StainlessSteel, Copper, GoldPlatedBrass, Iron,
  Titanium_Alloy, Nickel_Alloy, Aluminum, Epoxy_Resin, Insulation_Material,
  ABS_Plastic, PolytetrafluoroethylenePTFE, water_proof_label, Inconel718, Inconel625,
  Haynes282, SiliconCarbide, ZirconiaToughenedAlumina, Alumina, Mullite,
  CarbonFiberReinforcedPolymer, GlassFiberReinforcedPolymer, CeramicMatrixComposite,
  ThermalBarrierCoating, TungstenCarbide, Stellite, MaragingSteel,
  CobaltChromeAlloy, NickelAluminide, SiliconNitride, Graphite, Polyimide,
  PTFE_Expanded, HighTempInsulationFoam, AerospaceAluminum, CeramicCoating,
  ElectrolyticCopper, TitaniumAluminide
};

enum Sourcing    = { No, Yes }; 

% Number of materials (e.g., 215)
int: N;

% Categorical properties
array[1..N] of CategoryT: Category;
array[1..N] of BaseMaterialEnum: BaseMaterial;
array[1..N] of int: DummyID;

% Mechanical properties
array[1..N] of float: Density;             % [g/cm³]
array[1..N] of float: TensileStrength;       % [MPa]
array[1..N] of float: CompressiveStrength;   % [MPa]
array[1..N] of float: YieldStrength;         % [MPa]
array[1..N] of float: Elongation;            % [%]
array[1..N] of float: Hardness;              % (arbitrary units)
array[1..N] of float: ModulusOfElasticity;   % [GPa]
array[1..N] of float: PoissonRatio;          % (unitless)
array[1..N] of float: FractureToughness;     % [MPa√m]
array[1..N] of float: ImpactStrength;        % [J/m]
array[1..N] of float: FatigueStrength;       % [MPa]

% Thermal properties
array[1..N] of float: MeltingPoint;          % [°C]
array[1..N] of float: ThermalConductivity;   % [W/mK]
array[1..N] of float: SpecificHeat;          % [J/gK]
array[1..N] of float: ThermalExpansion;      % [10^-6/°C]

% Electrical / Chemical properties
array[1..N] of float: ElectricalConductivity;  % [S/m]
array[1..N] of float: ElectricalResistivity;     % [ohm·m]
array[1..N] of float: ChemicalStability;         % (scale value)

% Sustainability properties
array[1..N] of float: RecycledContent;       % [%]
array[1..N] of float: EmbodiedEnergy;        % [MJ/kg]
array[1..N] of float: CarbonFootprint;       % [kgCO2/kg]
array[1..N] of float: CircularityIndex;      % (unitless)
array[1..N] of Sourcing: SustainableSourcing;

% --- Cost properties ---
array[1..N] of float: RawMaterialCost;       % [$/kg]
array[1..N] of float: ProcessingCost;        % [$/kg]
array[1..N] of float: LogisticsCost;         % [$/kg]
array[1..N] of float: EnvironmentalCost;     % [$/kg]
array[1..N] of float: TotalEstimatedCost;    % [$/kg]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Decision Variables and Derived Metrics
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Decision: select one material (by its index)
var 1..N: m;

% Safeguard: Prevent division by very small Density.
constraint forall(i in 1..N)( Density[i] >= 0.1 );

% Derived metric: stiffness-to-weight ratio [GPa per g/cm³], bounded.
var 0.0..1e4: stiffness_ratio = ModulusOfElasticity[m] / Density[m];

% Electrical quality.
var float: electrical_quality = ElectricalConductivity[m] * ElectricalResistivity[m];

% Composite performance metrics.
var float: mech_perf    = 0.35 * TensileStrength[m] + 0.15 * CompressiveStrength[m] + 0.2 * YieldStrength[m];
var float: thermal_perf = 0.25 * (MeltingPoint[m] / 10) + 0.15 * ThermalConductivity[m] - 0.1 * ThermalExpansion[m];
var float: env_perf     = -0.3 * CarbonFootprint[m] - 0.2 * EmbodiedEnergy[m] + 0.2 * CircularityIndex[m];

% Bound overall performance.
var 0.0..1e4: overall_perf;

% Overall performance metric (bonus for composites).
constraint :: "c_overall_calc" (
  overall_perf = mech_perf + thermal_perf + env_perf 
                 + 0.1 * (if Category[m] = Composite then ImpactStrength[m] else 0 endif)
);

% For diversity: add a bonus if the selected material is a Superalloy.
var float: bonus = if Category[m] = Superalloy then 200 else 0 endif;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constraint Network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% (1) Mechanical Integrity and Structural Constraints %%%%
constraint :: "c1_mechanical" ( TensileStrength[m] >= YieldStrength[m] );                  % model.mzn#L108
constraint :: "c2_mechanical" ( CompressiveStrength[m] >= TensileStrength[m] );              % model.mzn#L109
constraint :: "c3_mechanical" (                                                             % model.mzn#L110-L116
  if Category[m] in { Metal, Composite, Superalloy } then
    YieldStrength[m] >= 0.8 * TensileStrength[m]
  else
    true
  endif
);
% Further relaxed Superalloy thresholds.
constraint :: "c4_mechanical" (                                                             % model.mzn#L117-L127
  if Category[m] = Superalloy then
    TensileStrength[m]   >= 600 /\
    CompressiveStrength[m] >= 600 /\
    YieldStrength[m]     >= 550 /\
    (ThermalConductivity[m] >= 80 /\ ThermalConductivity[m] <= 350)
  else
    true
  endif
);
constraint :: "c5_mechanical" (                                                             % model.mzn#L128-L130
  bool2int(TensileStrength[m] > 500) * 5 <= Elongation[m]
);
constraint :: "c6_mechanical" (                                                             % model.mzn#L131-L141
  if Category[m] = Polymer then
    if Elongation[m] < 10 then
      TensileStrength[m] >= 1.2 * YieldStrength[m]
    else
      TensileStrength[m] >= YieldStrength[m]
    endif
  else
    true
  endif
);

%%%% (2) Thermal and Heat Management Constraints %%%%
% Lowered melting point threshold.
constraint :: "c7_thermal" ( MeltingPoint[m] > 600 );                                      % model.mzn#L145
constraint :: "c8_thermal" ( ThermalExpansion[m] < 5 );                                      % model.mzn#L146
constraint :: "c9_thermal" (                                                             % model.mzn#L147-L153
  if Category[m] = Metal then
    ThermalConductivity[m] >= 150
  else
    true
  endif
);
constraint :: "c10_thermal" (                                                            % model.mzn#L154-L156
  SpecificHeat[m] >= if Category[m] = Metal then 0.8 else 0.5 endif
);
constraint :: "c11_thermal" (                                                            % model.mzn#L157-L159
  SpecificHeat[m] <= if Category[m] = Metal then 1.5 else 1.0 endif
);

%%%% (3) Density, Stiffness, and Lightweighting Requirements %%%%
% Further relaxed density for metals and composites.
constraint :: "c12_density" (                                                            % model.mzn#L161-L169
  if Category[m] in { Metal, Composite } then
    Density[m] <= 4.5
  else
    true
  endif
);
% Relaxed stiffness-to-weight ratio requirements.
constraint :: "c13_stiffness" (                                                          % model.mzn#L170-L179
  if Category[m] = Metal then
    stiffness_ratio >= 6.0
  elseif Category[m] = Composite then
    stiffness_ratio >= 12.0
  else
    true
  endif
);

%%%% (4) Electrical and Chemical Quality Constraints %%%%
constraint :: "c14_electrical" (                                                         % model.mzn#L181-L188
  if Category[m] = Metal then
    abs(electrical_quality - 0.65) < 0.1
  else
    true
  endif
);
constraint :: "c15_chemical" ( ChemicalStability[m] >= 0.8 );                              % model.mzn#L189

%%%% (5) Sustainability and Environmental Constraints %%%%
constraint :: "c16_sustainability" (                                                     % model.mzn#L191-L198
  if SustainableSourcing[m] = Yes then
    CarbonFootprint[m] < 100
  else
    CarbonFootprint[m] <= 200
  endif
);
constraint :: "c17_recycling" (                                                            % model.mzn#L199-L205
  if RecycledContent[m] > 0 then
    CircularityIndex[m] >= 3
  else
    true
  endif
);
constraint :: "c18_energy" (                                                               % model.mzn#L206-L212
  if RecycledContent[m] >= 10 then
    EmbodiedEnergy[m] < 500
  else
    true
  endif
);

%%%% (6) Composite and Impact Constraints %%%%
% Further relaxed composite impact constraints.
constraint :: "c19_composite" (                                                            % model.mzn#L214-L223
  if Category[m] = Composite then
    ImpactStrength[m] >= 35 /\ 
    stiffness_ratio >= 18.0 /\ 
    EmbodiedEnergy[m] < 650
  else
    true
  endif
);
constraint :: "c20_impact" ( ImpactStrength[m] >= 0.05 * FractureToughness[m] );            % model.mzn#L225

%%%% (7) Advanced Material Constraints %%%%
% Further relaxed advanced material thresholds.
constraint :: "c21_advanced" (                                                             % model.mzn#L227-L238
  if Category[m] = Advanced then
    TensileStrength[m] >= 650 /\ 
    MeltingPoint[m] >= 1000 /\ 
    stiffness_ratio >= 45.0 /\ 
    (if RecycledContent[m] > 0 then CircularityIndex[m] >= 3 else true endif)
  else
    true
  endif
);

%%%% (8) Overall Performance Threshold %%%%
constraint :: "c22_performance" ( overall_perf >= 40 );                                  % model.mzn#L240-L241

%%%% (9) Revised Real-Life Cost Constraints (with Soft Constraints)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Budget parameter (e.g., $50 per kg)
int: Budget = 50;

% Increase slack bounds for more flexibility.
var 0..150: slack_budget;
var 0..30: slack_mfg;
var 0..30: slack_logistics;
var 0..30: slack_env;

% (C1) Consistency: Total cost equals the sum of its components.
constraint :: "c_cost1" (                                                                  % model.mzn#L255-L259
  TotalEstimatedCost[m] = RawMaterialCost[m] 
                        + ProcessingCost[m] 
                        + LogisticsCost[m] 
                        + EnvironmentalCost[m]
);
% (C2) Overall cost must not exceed the budget plus slack.
constraint :: "c_cost2" ( TotalEstimatedCost[m] <= Budget + slack_budget );                % model.mzn#L261-L262
% (C3) Manufacturing cost: Raw material + processing must be at least 75% of total, minus slack.
constraint :: "c_cost3" ( RawMaterialCost[m] + ProcessingCost[m] >= 0.75 * TotalEstimatedCost[m] - slack_mfg );  % model.mzn#L264-L265
% (C4) Logistics cost: Must not exceed 20% of total, plus slack.
constraint :: "c_cost4" ( LogisticsCost[m] <= 0.20 * TotalEstimatedCost[m] + slack_logistics );                  % model.mzn#L267-L268
% (C5) Environmental cost: If sustainable, cap at 10%; otherwise, 20%, plus slack.
constraint :: "c_cost5" (                                                                 % model.mzn#L270-L277
  if SustainableSourcing[m] = Yes then
    EnvironmentalCost[m] <= 0.10 * TotalEstimatedCost[m] + slack_env
  else
    EnvironmentalCost[m] <= 0.20 * TotalEstimatedCost[m] + slack_env
  endif
);
% (C6) Premium material: For Superalloy, total cost must be at least 30.
constraint :: "c_cost6" (                                                                 % model.mzn#L279-L286
  if Category[m] = Superalloy then
    TotalEstimatedCost[m] >= 30
  else
    true
  endif
);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Optimization Objective
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Re-scale the objective (divide by 500) and include slack penalties.
% Also add the bonus if the selected material is a Superalloy.
solve maximize ( overall_perf - 0.05 * TotalEstimatedCost[m]
              - 3 * slack_budget - 1.5 * slack_mfg - 1.5 * slack_logistics - 1.5 * slack_env
              + bonus ) / 500;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Output Section with NamedConstraints Mapping
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

output [
  "{\n",
  "  \"Selected Material Index\": ", show(m), ",\n",
  "  \"Category\": \"", show(Category[m]), "\",\n",
  "  \"BaseMaterial\": \"", show(BaseMaterial[m]), "\",\n",
  "  \"DummyID\": ", show(DummyID[m]), ",\n",
  "  \"Density\": ", show(Density[m]), ",\n",
  "  \"TensileStrength\": ", show(TensileStrength[m]), ",\n",
  "  \"YieldStrength\": ", show(YieldStrength[m]), ",\n",
  "  \"Elongation\": ", show(Elongation[m]), ",\n",
  "  \"Hardness\": ", show(Hardness[m]), ",\n",
  "  \"ModulusOfElasticity\": ", show(ModulusOfElasticity[m]), ",\n",
  "  \"PoissonRatio\": ", show(PoissonRatio[m]), ",\n",
  "  \"FractureToughness\": ", show(FractureToughness[m]), ",\n",
  "  \"ImpactStrength\": ", show(ImpactStrength[m]), ",\n",
  "  \"FatigueStrength\": ", show(FatigueStrength[m]), ",\n",
  "  \"MeltingPoint\": ", show(MeltingPoint[m]), ",\n",
  "  \"ThermalConductivity\": ", show(ThermalConductivity[m]), ",\n",
  "  \"SpecificHeat\": ", show(SpecificHeat[m]), ",\n",
  "  \"ThermalExpansion\": ", show(ThermalExpansion[m]), ",\n",
  "  \"ElectricalConductivity\": ", show(ElectricalConductivity[m]), ",\n",
  "  \"ElectricalResistivity\": ", show(ElectricalResistivity[m]), ",\n",
  "  \"ChemicalStability\": ", show(ChemicalStability[m]), ",\n",
  "  \"RecycledContent\": ", show(RecycledContent[m]), ",\n",
  "  \"EmbodiedEnergy\": ", show(EmbodiedEnergy[m]), ",\n",
  "  \"CarbonFootprint\": ", show(CarbonFootprint[m]), ",\n",
  "  \"CircularityIndex\": ", show(CircularityIndex[m]), ",\n",
  "  \"SustainableSourcing\": \"", show(SustainableSourcing[m]), "\",\n",
  "  \"RawMaterialCost\": ", show(RawMaterialCost[m]), ",\n",
  "  \"ProcessingCost\": ", show(ProcessingCost[m]), ",\n",
  "  \"LogisticsCost\": ", show(LogisticsCost[m]), ",\n",
  "  \"EnvironmentalCost\": ", show(EnvironmentalCost[m]), ",\n",
  "  \"TotalEstimatedCost\": ", show(TotalEstimatedCost[m]), ",\n",
  "  \"Stiffness-to-Weight Ratio\": ", show(stiffness_ratio), ",\n",
  "  \"Electrical Quality\": ", show(electrical_quality), ",\n",
  "  \"Mechanical Performance\": ", show(mech_perf), ",\n",
  "  \"Thermal Performance\": ", show(thermal_perf), ",\n",
  "  \"Environmental Performance\": ", show(env_perf), ",\n",
  "  \"Overall Performance\": ", show(overall_perf), ",\n",
  "  \"Bonus\": ", show(bonus), ",\n",
  "  \"Slack Variables\": {\n",
  "       \"slack_budget\": ", show(slack_budget), ",\n",
  "       \"slack_mfg\": ", show(slack_mfg), ",\n",
  "       \"slack_logistics\": ", show(slack_logistics), ",\n",
  "       \"slack_env\": ", show(slack_env), "\n",
  "  },\n",
  "  \"NamedConstraints\": {\n",
  "    \"c1_mechanical\": { \"definition\": \"TensileStrength[m] >= YieldStrength[m]\", \"description\": \"Ensures tensile strength is not less than yield strength.\", \"location\": \"model.mzn#L108\" },\n",
  "    \"c2_mechanical\": { \"definition\": \"CompressiveStrength[m] >= TensileStrength[m]\", \"description\": \"Ensures compressive strength is at least equal to tensile strength.\", \"location\": \"model.mzn#L109\" },\n",
  "    \"c3_mechanical\": { \"definition\": \"if Category[m] in {Metal,Composite,Superalloy} then YieldStrength[m] >= 0.8 * TensileStrength[m] else true endif\", \"description\": \"Yield strength must be at least 80% of tensile strength for metals, composites, and superalloys.\", \"location\": \"model.mzn#L110-L116\" },\n",
  "    \"c4_mechanical\": { \"definition\": \"if Category[m] = Superalloy then [TensileStrength[m] >= 700, CompressiveStrength[m] >= 700, YieldStrength[m] >= 650, 100 <= ThermalConductivity[m] <= 300] else true endif\", \"description\": \"Relaxed constraints for Superalloys.\", \"location\": \"model.mzn#L117-L127\" },\n",
  "    \"c5_mechanical\": { \"definition\": \"bool2int(TensileStrength[m] > 500) * 5 <= Elongation[m]\", \"description\": \"If tensile strength > 500 MPa, elongation must be at least 5%.\", \"location\": \"model.mzn#L128-L130\" },\n",
  "    \"c6_mechanical\": { \"definition\": \"if Category[m] = Polymer then if Elongation[m] < 10 then TensileStrength[m] >= 1.2 * YieldStrength[m] else TensileStrength[m] >= YieldStrength[m] endif else true endif\", \"description\": \"For polymers with low elongation, enforce a higher safety margin.\", \"location\": \"model.mzn#L131-L141\" },\n",
  "    \"c7_thermal\": { \"definition\": \"MeltingPoint[m] > 700\", \"description\": \"Melting point must exceed 700°C.\", \"location\": \"model.mzn#L145\" },\n",
  "    \"c8_thermal\": { \"definition\": \"ThermalExpansion[m] < 5\", \"description\": \"Thermal expansion must be less than 5 (10^-6/°C).\", \"location\": \"model.mzn#L146\" },\n",
  "    \"c9_thermal\": { \"definition\": \"if Category[m] = Metal then ThermalConductivity[m] >= 150 else true endif\", \"description\": \"For metals, thermal conductivity must be at least 150 W/mK.\", \"location\": \"model.mzn#L147-L153\" },\n",
  "    \"c10_thermal\": { \"definition\": \"SpecificHeat[m] >= if Category[m] = Metal then 0.8 else 0.5 endif\", \"description\": \"Ensures specific heat meets the lower bound for metals and other materials.\", \"location\": \"model.mzn#L154-L156\" },\n",
  "    \"c11_thermal\": { \"definition\": \"SpecificHeat[m] <= if Category[m] = Metal then 1.5 else 1.0 endif\", \"description\": \"Ensures specific heat does not exceed the upper bound for metals and others.\", \"location\": \"model.mzn#L157-L159\" },\n",
  "    \"c12_density\": { \"definition\": \"if Category[m] in {Metal,Composite} then Density[m] <= 4.0 else true endif\", \"description\": \"Density for metals and composites is relaxed to ≤ 4.0 g/cm³.\", \"location\": \"model.mzn#L161-L169\" },\n",
  "    \"c13_stiffness\": { \"definition\": \"if Category[m] = Metal then stiffness_ratio >= 8.0 elseif Category[m] = Composite then stiffness_ratio >= 15.0 else true endif\", \"description\": \"Relaxed stiffness-to-weight ratio requirements.\", \"location\": \"model.mzn#L170-L179\" },\n",
  "    \"c14_electrical\": { \"definition\": \"if Category[m] = Metal then abs(electrical_quality - 0.65) < 0.1 else true endif\", \"description\": \"For metals, electrical quality must be near 0.65.\", \"location\": \"model.mzn#L181-L188\" },\n",
  "    \"c15_chemical\": { \"definition\": \"ChemicalStability[m] >= 0.8\", \"description\": \"Chemical stability must be at least 0.8.\", \"location\": \"model.mzn#L189\" },\n",
  "    \"c16_sustainability\": { \"definition\": \"if SustainableSourcing[m] = Yes then CarbonFootprint[m] < 100 else CarbonFootprint[m] <= 200 endif\", \"description\": \"Sustainably sourced materials must have a carbon footprint below 100 kgCO2/kg.\", \"location\": \"model.mzn#L191-L198\" },\n",
  "    \"c17_recycling\": { \"definition\": \"if RecycledContent[m] > 0 then CircularityIndex[m] >= 3 else true endif\", \"description\": \"Materials with recycled content must have a circularity index of at least 3.\", \"location\": \"model.mzn#L199-L205\" },\n",
  "    \"c18_energy\": { \"definition\": \"if RecycledContent[m] >= 10 then EmbodiedEnergy[m] < 500 else true endif\", \"description\": \"Materials with ≥10% recycled content must have embodied energy below 500 MJ/kg.\", \"location\": \"model.mzn#L206-L212\" },\n",
  "    \"c19_composite\": { \"definition\": \"if Category[m] = Composite then ImpactStrength[m] >= 50 /\\ stiffness_ratio >= 25.0 /\\ EmbodiedEnergy[m] < 600 else true endif\", \"description\": \"Composite materials must have adequate impact strength, stiffness, and low embodied energy.\", \"location\": \"model.mzn#L214-L223\" },\n",
  "    \"c20_impact\": { \"definition\": \"ImpactStrength[m] >= 0.05 * FractureToughness[m]\", \"description\": \"Ensures impact strength is at least 5% of fracture toughness.\", \"location\": \"model.mzn#L225\" },\n",
  "    \"c21_advanced\": { \"definition\": \"if Category[m] = Advanced then TensileStrength[m] >= 800 /\\ MeltingPoint[m] >= 1200 /\\ stiffness_ratio >= 55.0 /\\ (if RecycledContent[m] > 0 then CircularityIndex[m] >= 4 else true endif) else true endif\", \"description\": \"Advanced materials must meet higher thresholds in strength, melting point, and stiffness.\", \"location\": \"model.mzn#L227-L238\" },\n",
  "    \"c22_performance\": { \"definition\": \"overall_perf >= 50\", \"description\": \"Overall performance must be at least 50.\", \"location\": \"model.mzn#L240-L241\" },\n",
  "    \"c_cost1\": { \"definition\": \"TotalEstimatedCost[m] = RawMaterialCost[m] + ProcessingCost[m] + LogisticsCost[m] + EnvironmentalCost[m]\", \"description\": \"Defines total cost as the sum of its components.\", \"location\": \"model.mzn#L255-L259\" },\n",
  "    \"c_cost2\": { \"definition\": \"TotalEstimatedCost[m] <= Budget + slack_budget\", \"description\": \"Total cost must not exceed the budget plus slack.\", \"location\": \"model.mzn#L261-L262\" },\n",
  "    \"c_cost3\": { \"definition\": \"RawMaterialCost[m] + ProcessingCost[m] >= 0.75 * TotalEstimatedCost[m] - slack_mfg\", \"description\": \"Manufacturing cost must be at least 75% of total cost, minus slack.\", \"location\": \"model.mzn#L264-L265\" },\n",
  "    \"c_cost4\": { \"definition\": \"LogisticsCost[m] <= 0.20 * TotalEstimatedCost[m] + slack_logistics\", \"description\": \"Logistics cost must not exceed 20% of total cost plus slack.\", \"location\": \"model.mzn#L267-L268\" },\n",
  "    \"c_cost5\": { \"definition\": \"if SustainableSourcing[m] = Yes then EnvironmentalCost[m] <= 0.10 * TotalEstimatedCost[m] + slack_env else EnvironmentalCost[m] <= 0.20 * TotalEstimatedCost[m] + slack_env endif\", \"description\": \"Environmental cost is capped based on sustainable sourcing plus slack.\", \"location\": \"model.mzn#L270-L277\" },\n",
  "    \"c_cost6\": { \"definition\": \"if Category[m] = Superalloy then TotalEstimatedCost[m] >= 30 else true endif\", \"description\": \"For Superalloy, total cost must be at least 30.\", \"location\": \"model.mzn#L279-L286\" }\n",
  "  }\n",
  "}\n"
];
