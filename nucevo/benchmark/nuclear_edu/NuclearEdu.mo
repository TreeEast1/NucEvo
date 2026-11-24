package NuclearEdu "Educational lumped-parameter nuclear thermal-hydraulic block library (causal connectors)"
  // All quantities SI. Deliberately small and fast to compile (~0.3-1 s per model).
  // Every component is a causal block: inputs are connectors `In`, outputs `Out`.

  connector Sig "causal signal connector"
    Real u;
  end Sig;

  model Const "constant signal source (frozen state so the value is observable)"
    parameter Real k = 1;
    output Sig y;
    Real s(start = k, fixed = true) "held constant";
  equation
    der(s) = 0;
    y.u = s;
  end Const;

  model PowerSource "imposed fission power"
    parameter Real P0 = 3.0e9 "thermal power [W]";
    output Sig P;
  equation
    P.u = P0;
  end PowerSource;

  model FuelPin "fuel + clad lumped heat transfer with gap conductance"
    parameter Real Cf = 1.0e8 "fuel heat capacity [J/K]";
    parameter Real Cc = 4.0e7 "clad heat capacity [J/K]";
    parameter Real G_gap = 1.0e7 "fuel-clad conductance [W/K]";
    parameter Real G_cl = 1.5e8 "clad-coolant conductance [W/K]";
    parameter Real Tf0 = 901 "initial fuel temperature [K]";
    parameter Real Tc0 = 601 "initial clad temperature [K]";
    input Sig Q_fission "fission heat [W]";
    input Sig Tcool "coolant temperature seen by the clad [K]";
    output Sig Tf;
    output Sig Tc;
    output Sig Qcool "heat delivered to the coolant [W]";
    Real Tf_s(start = Tf0, fixed = true);
    Real Tc_s(start = Tc0, fixed = true);
  equation
    Cf*der(Tf_s) = Q_fission.u - G_gap*(Tf_s - Tc_s);
    Cc*der(Tc_s) = G_gap*(Tf_s - Tc_s) - G_cl*(Tc_s - Tcool.u);
    Qcool.u = G_cl*(Tc_s - Tcool.u);
    Tf.u = Tf_s;
    Tc.u = Tc_s;
  end FuelPin;

  model Channel "core coolant node: advection + stored energy"
    parameter Real Cp = 5200 "coolant specific heat [J/(kg.K)]";
    parameter Real Cth = 4.0e7 "coolant thermal capacitance [J/K]";
    parameter Real Tin0 = 565 "initial inlet temperature [K]";
    parameter Real Tout0 = 597 "initial outlet temperature [K]";
    input Sig Tin;
    input Sig mdot "mass flow [kg/s]";
    input Sig Q_fuel "heat from fuel [W]";
    output Sig Tout;
    output Sig Tavg;
    Real T(start = Tout0, fixed = true);
  equation
    Cth*der(T) = Q_fuel.u + mdot.u*Cp*(Tin.u - T);
    Tout.u = T;
    Tavg.u = 0.5*(Tin.u + T);
  end Channel;

  model SteamGenerator "condensing secondary, effectiveness-NTU, dynamic primary outlet"
    parameter Real Cp = 5200 "primary Cp [J/(kg.K)]";
    parameter Real UA = 3.33e7 "overall conductance [W/K]";
    parameter Real Tsec = 490 "secondary saturation temperature [K]";
    parameter Real Cth = 4.0e7 "SG primary-side thermal capacitance [J/K]";
    parameter Real Tin0 = 565 "initial primary outlet temperature [K]";
    input Sig Tin "hot-leg inlet temperature [K]";
    input Sig mdot "primary mass flow [kg/s]";
    output Sig Tout "cold-leg outlet temperature [K]";
    output Sig Qsg "heat removed by the secondary [W]";
    Real T(start = Tin0, fixed = true);
    Real eps;
  equation
    eps = 1 - exp(-UA/(Cp*max(mdot.u, 1.0)));
    Qsg.u = eps*Cp*mdot.u*(Tin.u - Tsec);
    Cth*der(T) = mdot.u*Cp*(Tin.u - T) - Qsg.u;
    Tout.u = T;
  end SteamGenerator;

  model Pump "centrifugal pump curve, explicit flow from circuit pressure drop"
    parameter Real Head0 = 3.2e6 "shutoff head [Pa]";
    parameter Real Mdot0 = 25000 "flow at zero head [kg/s]";
    input Sig dp "circuit pressure drop [Pa]";
    output Sig mdot "mass flow [kg/s]";
  equation
    mdot.u = Mdot0*sqrt(max(0.0, 1.0 - dp.u/Head0));
  end Pump;

  model Pipe "quadratic hydraulic resistance"
    parameter Real K = 4.756e-3 "resistance [Pa/(kg/s)^2]";
    input Sig mdot;
    output Sig dp;
  equation
    dp.u = K*mdot.u*abs(mdot.u);
  end Pipe;

  model PointKinetics "point kinetics with fuel and moderator feedback"
    parameter Real Beta = 0.0065 "delayed neutron fraction";
    parameter Real L = 1.0e-5 "prompt neutron lifetime [s]";
    parameter Real Lambda = 0.0784 "precursor decay constant [1/s]";
    parameter Real Alpha_f = -2.5e-5 "fuel temperature coefficient [1/K]";
    parameter Real Alpha_m = -5.0e-4 "moderator temperature coefficient [1/K]";
    parameter Real Tf0 = 901 "reference fuel temperature [K]";
    parameter Real Tm0 = 581 "reference moderator temperature [K]";
    parameter Real P0 = 3.0e9 "rated thermal power [W]";
    input Sig rho_ext "externally inserted reactivity";
    input Sig Tf "fuel temperature [K]";
    input Sig Tm "moderator temperature [K]";
    output Sig P "thermal power [W]";
    output Sig n "normalized neutron density";
    Real nn(start = 1, fixed = true);
    Real c(start = Beta/(L*Lambda), fixed = true);
    Real rho;
  equation
    rho = rho_ext.u + Alpha_f*(Tf.u - Tf0) + Alpha_m*(Tm.u - Tm0);
    der(nn) = (rho - Beta)/L*nn + Lambda*c;
    der(c) = Beta/L*nn - Lambda*c;
    P.u = P0*nn;
    n.u = nn;
  end PointKinetics;

  model Mixer "two-stream adiabatic mixer"
    parameter Real Cp = 5200;
    input Sig T1;
    input Sig m1;
    input Sig T2;
    input Sig m2;
    output Sig T;
    output Sig m;
  equation
    m.u = m1.u + m2.u;
    T.u = (m1.u*T1.u + m2.u*T2.u)/max(m.u, 1.0e-6);
  end Mixer;

  model Splitter "flow split with fixed fraction"
    parameter Real frac = 0.5 "fraction to leg 1";
    input Sig T;
    input Sig m;
    output Sig T1;
    output Sig m1;
    output Sig T2;
    output Sig m2;
  equation
    m1.u = frac*m.u;
    m2.u = (1 - frac)*m.u;
    T1.u = T.u;
    T2.u = T.u;
  end Splitter;

  model Controller "proportional heater controller with saturation"
    parameter Real Kp = 1.0e8 "gain [W/K]";
    parameter Real Tset = 600 "setpoint [K]";
    parameter Real Qmax = 4.5e9 "saturation [W]";
    input Sig T;
    output Sig Q;
  equation
    Q.u = min(Qmax, max(0.0, Kp*(Tset - T.u)));
  end Controller;

  model Sensor "passthrough measurement block"
    input Sig y_in;
    output Sig y;
  equation
    y.u = y_in.u;
  end Sensor;

end NuclearEdu;
