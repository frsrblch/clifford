use clifford::pga_3d::*;
use std::ops::*;

#[test]
fn rotor_sqrt() {
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

    assert_eq!(v_90, rot_90 >> v);
}

#[test]
fn motor_from_scalar() {
    let one = Scalar { s: 1. };
    let motor = Motor::from(one);
    assert_eq!(motor, Motor { s: 1., ..zero() });
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
    assert!((v1 - v2).norm2() < 1e-10);
}

#[test]
fn unit_motor_has_norm_1() {
    use rand::{thread_rng, Rng};
    let mut rng = thread_rng();
    for _ in 0..100 {
        let m = rng.gen::<Unit<Motor<f64>>>();
        assert!(m.value().norm2().to_f32().is_one());
    }
}

#[test]
fn point_constructor() {
    let (x, y, z) = (2., 3., 5.);
    let expected =
        Vector::new(1., 0., 0., -x) ^ Vector::new(0., 1., 0., -y) ^ Vector::new(0., 0., 1., -z);
    let actual = point(x, y, z).value();
    assert_eq!(expected, actual);
}

#[test]
fn non_unit_point_to_coordinates() {
    let (x, y, z) = (2f64, 3., 5.);
    let pt = point(x, y, z) * -2.;

    let (cx, cy, cz) = pt.coords();

    assert_eq!((x, y, z), (cx, cy, cz));
}

#[test]
fn unit_point_to_coordinates() {
    let (x, y, z) = (2f64, 3., 5.);
    let pt = point(x, y, z) * -2.;
    dbg!(pt, pt.unit());
    let pt = pt.unit();

    let (cx, cy, cz) = pt.coords();

    assert_eq!((x, y, z), (cx, cy, cz));
}

#[test]
fn plane_reflection() {
    let pt = point(1., 2., 3.);

    let xy = Vector::new(0., 0., 1., 0.);

    let reflected = xy.sandwich(pt);
    let expected = point(1., 2., -3.);

    assert_eq!(expected, reflected);
}

#[test]
fn line_rotation() {
    let pt = point(1., 2., 3.);

    let xy = Vector::new(0., 0., 1., 0.);
    let yz = Vector::new(1., 0., 0., 0.);
    let line = xy ^ yz;

    let reflected = line.sandwich(pt);
    let expected = point(-1., 2., -3.);

    assert_eq!(expected, reflected);
}

#[test]
fn line_rotation_by_angle() {
    let (x, y, z) = (2., 3., 5.);
    let pt = point(x, y, z).value();

    let xy = Vector::new(0., 0., 1., 0.);
    let yz = Vector::new(1., 0., 0., 0.);
    let line = xy ^ yz;
    let angle = Scalar::<f64>::FRAC_PI_4();
    let (sin, cos) = angle.sin_cos();
    let motor = line.unit() * sin + cos;

    let rotated = motor >> pt;
    let expected = point(-z, y, x).value();

    assert!((expected - rotated).norm2() < f64::epsilon());
}

#[test]
fn line_rotation_by_angle_sqrt() {
    let (x, y, z) = (2., 3., 5.);
    let pt = point(x, y, z).value();

    let xy = Vector::new(0., 0., 1., 0.);
    let yz = Vector::new(1., 0., 0., 0.);
    let line = xy ^ yz;
    let angle = Scalar::<f64>::FRAC_PI_2();
    let (sin, cos) = angle.sin_cos();
    let motor = (line.unit() * sin + cos).sqrt();

    let rotated = motor.sandwich(pt);
    let expected = point(-z, y, x).value();

    assert!((expected - rotated).norm2() < f64::epsilon());
}

#[test]
fn point_translation() {
    let (x, y, z) = (2., 3., 5.);
    let pt = point(x, y, z);

    let xy = Vector::new(0., 0., 1., 0.);
    let xy_offset = Vector::new(0., 0., 1., 0.5);

    let translator = xy * xy_offset;

    let expected = point(x, y, z + 1.);

    assert_eq!(expected, translator.sandwich(pt));
}

#[test]
fn point_translation_sqrt() {
    let (x, y, z) = (2., 3., 5.);
    let pt = point(x, y, z);

    let xy = Vector::new(0., 0., 1., 0.);
    let xy_offset = Vector::new(0., 0., 1., 1.);

    let translator = (xy * xy_offset).sqrt();

    let expected = point(x, y, z + 1.);

    assert_eq!(expected, translator.sandwich(pt));
}

#[test]
fn translator_from_points() {
    let (x, y, z) = (2., 3., 5.);
    let (a, b, c) = (1., 2., 3.);
    let pt = point(x, y, z);

    let origin = point(0., 0., 0.).value();
    let offset = point(a * 0.5, b * 0.5, c * 0.5).value();
    let translator = offset * origin;

    let expected = point(x + a, y + b, z + c);

    assert_eq!(expected, translator.sandwich(pt));
}

#[test]
fn translator_from_points_sqrt() {
    let (x, y, z) = (2., 3., 5.);
    let (a, b, c) = (1., 2., 3.);
    let pt = point(x, y, z);

    let origin = point(0f64, 0., 0.).value();
    let offset = point(a, b, c).value();
    let translator2 = offset * origin;
    let translator = translator2.sqrt();
    let translator_log = translator2.log().mul(0.5).exp();

    let expected = point(x + a, y + b, z + c);
    assert_eq!(expected, translator.sandwich(pt));
    assert_eq!(translator2, -translator * translator);
    assert_eq!(translator, translator_log);
}

#[test]
fn rotation_around_line_log_and_exp() {
    let xy = Vector::new(0., 0., 1., 0.);
    let yz = Vector::new(1., 0., 0., 0.);
    let line = xy ^ yz;
    let angle = Scalar::<f64>::FRAC_PI_4();
    let (sin, cos) = angle.sin_cos();
    let motor = line.unit() * sin + cos;

    let expected = motor.sqrt();
    let actual = motor.log().mul(0.5).exp();

    assert_eq!(expected, actual);
}

#[test]
fn plane_translation_log_exp() {
    let xy = Vector::new(0f64, 0., 1., 0.);
    let offset = Vector { w: 2., ..xy };

    let motor = offset * xy.inv();

    assert_eq!(motor.sqrt(), motor.log().mul(0.5).exp());
}

#[test]
fn point_translation_log_exp() {
    let p0 = point(0f64, 0., 0.).value();
    let p1 = point(1., 2., 3.).value();

    let motor = p1 * p0;

    assert_eq!(motor.sqrt(), motor.log().mul(0.5).exp());
    assert_eq!(p1, motor.sqrt() >> p0);
}
