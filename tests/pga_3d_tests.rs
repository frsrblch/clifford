use clifford::pga3::*;

#[test]
fn rotor_sqrt() {
    use num_sqrt::Sqrt;

    let rot_180 = Motor {
        s: 0.,
        xy: -1.,
        ..Default::default()
    };
    let v = Vector {
        x: 2.,
        y: 3.,
        z: 5.,
        w: 1.,
    };
    let v_90 = Vector {
        x: -3.,
        y: 2.,
        z: 5.,
        w: 1.,
    };

    let rot_90 = rot_180.sqrt();
    dbg!(rot_90, rot_90.sandwich(v));

    assert_eq!(v_90, rot_90.sandwich(v));
    // panic!("done");
}

#[test]
fn motor_from_scalar() {
    let s = Scalar { s: 1. };
    let _m = Motor::from(s);
}

#[test]
fn rotation_and_unit_rotation() {
    let (sin, cos) = std::f64::consts::FRAC_PI_4.sin_cos();
    let motor = Motor::new(sin, cos, 0., 0., 0., 0., 0., 0.) * 2.;
    let unit = motor.unit();

    let v = Vector::new(1., 2., 3., 0.);

    let v1 = motor.sandwich(v);
    let v2 = unit.sandwich(v); // interestingly, the sqrt adds inaccuracy

    dbg!(v1, v2);
    assert!((v1 - v2).norm2().s < 1e-10);
    // panic!();
}

#[test]
fn f0_ops() {
    use f_zero::f0;
    let vector: Value<f0> = Value::Vector(Default::default());
    let bivector: Value<f0> = Value::Bivector(Default::default());
    assert_eq!(
        Some(Value::Flector(Default::default())),
        vector.geo(bivector)
    );
    assert_eq!(
        Some(Value::Vector(Default::default())),
        vector.dot(bivector)
    );
}
