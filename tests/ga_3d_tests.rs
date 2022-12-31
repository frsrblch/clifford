use clifford::ga_3d::*;

#[cfg(feature = "dyn")]
#[test]
fn value_geo() {
    let v = Value::Vector(Vector::new(1., 0., 0.));
    assert_eq!(
        Value::Motor(Motor {
            s: 1.,
            ..Motor::default()
        }),
        v.geo(v).unwrap()
    );
}

#[test]
fn vector_inv() {
    let v = Vector::new(2., 3., 5.);
    assert_eq!(v.inv(), 1. / v);
}

#[test]
fn scalar_antigeometric_product() {
    let s = Scalar::new(1.);
    let neg_e123 = s.antigeo(s);
    let e123 = neg_e123.rev();
    let s_ = e123.dual();
    assert_eq!(s_, s);
}

#[test]
fn differing_dual_definitions() {
    let i = Trivector::new(1.);
    let v = Vector::new(2., 3., 5.);

    let dual = |v: Vector<f32>| v.geo(i.inv());
    let dual1 = |v: Bivector<f32>| v.geo(i.inv());

    assert_eq!(dual1(dual(v)), -v); // according to LAGA
    assert_eq!(v.dual().dual(), v); // according to RGA wiki
}

#[test]
fn change_of_basis() {
    let b = Vector::new(2., 3., 5.);

    let a1 = Vector::new(1., 0., 0.);
    let a2 = Vector::new(1., 1., 0.);
    let a3 = Vector::new(1., 1., 1.);
    let a123 = a1 ^ a2 ^ a3;

    let alpha_1 = (b ^ a2 ^ a3) / a123;
    let alpha_2 = (a1 ^ b ^ a3) / a123;
    let alpha_3 = (a1 ^ a2 ^ b) / a123;
    dbg!(alpha_1, alpha_2, alpha_3);

    assert_eq!(b, alpha_1 * a1 + alpha_2 * a2 + alpha_3 * a3);
}

#[test]
fn f64_mul_vector() {
    let v = Vector::new(1., 2., 3.);
    let a = 2f64;

    assert_eq!(a * v, v * a);
}

#[test]
fn unit_motor_has_norm_1() {
    use rand::{thread_rng, Rng};
    let mut rng = thread_rng();
    for _ in 0..100 {
        let m = rng.gen::<Unit<Motor<f64>>>();
        assert_eq!(m.value().norm2().to_f32().s, 1.);
    }
}

#[test]
fn sandwich_shr_overload() {
    let m = Bivector {
        xy: 1.,
        ..Default::default()
    };
    let v = Vector {
        x: 1.,
        y: 2.,
        z: 3.,
    };

    let expected = Vector {
        x: -1.,
        y: -2.,
        z: 3.,
    };
    let actual = m >> v;

    assert_eq!(expected, actual);
}

#[test]
fn motor_log_and_angle() {
    let plane = Bivector::new(1., 2., 3.).unit();
    let angle = Scalar::new(std::f64::consts::FRAC_PI_3);
    let motor = Motor::from_plane_and_angle(plane, angle);

    assert_eq!(motor.log(), plane);
    assert_eq!(motor.angle(), angle);
}
