#[cfg(feature = "cga_3d")]
use clifford::cga_3d::*;

//! All references to Geometric Algebra for Computer Science

#[allow(dead_code)]
fn invert1(v: Vector<f64>) -> Vector<f64> {
    const E4: Vector<f64> = Vector::new(0., 0., 0., -1., 0.);
    -E4.geo(v).geo(E4).1
}

#[allow(dead_code)]
fn invert2(v: Bivector<f64>) -> Bivector<f64> {
    const E4: Vector<f64> = Vector::new(0., 0., 0., -1., 0.);
    -E4.geo(v).geo(E4).2
}

#[allow(dead_code)]
fn invert3(v: Trivector<f64>) -> Trivector<f64> {
    const E4: Vector<f64> = Vector::new(0., 0., 0., -1., 0.);
    -E4.geo(v).geo(E4).3
}

#[allow(dead_code)]
fn invert4(v: Quadvector<f64>) -> Quadvector<f64> {
    const E4: Vector<f64> = Vector::new(0., 0., 0., -1., 0.);
    -E4.geo(v).geo(E4).4
}

impl Inverse for Vector<f64> {
    fn inv(self) -> Vector<f64> {
        const E4: Vector<f64> = Vector::new(0., 0., 0., -1., 0.);
        -E4.geo(self).geo(E4).1
    }
}

impl Inverse for Bivector<f64> {
    fn inv(self) -> Bivector<f64> {
        const E4: Vector<f64> = Vector::new(0., 0., 0., -1., 0.);
        -E4.geo(self).geo(E4).2
    }
}

impl Inverse for Trivector<f64> {
    fn inv(self) -> Trivector<f64> {
        const E4: Vector<f64> = Vector::new(0., 0., 0., -1., 0.);
        -E4.geo(self).geo(E4).3
    }
}

impl Inverse for Quadvector<f64> {
    fn inv(self) -> Quadvector<f64> {
        const E4: Vector<f64> = Vector::new(0., 0., 0., -1., 0.);
        -E4.geo(self).geo(E4).4
    }
}

pub type Translator<T> = Multivector<T, Zero, Bivector<T>, Zero, Zero, Zero>;

pub fn translator(x: f64, y: f64, z: f64) -> (Translator<f64>, Translator<f64>) {
    let t = Vector::new(x, y, z, 0., 0.);
    let b = 0.5 * t * N_BAR;

    // this works because N_BAR * N_BAR = 0 for any t
    // so the reversed bivector sign is enough to make them cancel out
    (1. - b, 1. + b)
}

#[test]
#[allow(non_snake_case)]
fn translation_test() {
    let (t, t_inv) = translator(2., 3., 5.);
    let p = point(1., 2., 3.);

    assert_eq!((t * t_inv).0, 1.);
    assert_eq!((t * t_inv).2, Default::default());
    assert_eq!((t * t_inv).4, Default::default());

    let expected = point(3., 5., 8.);
    let actual = (t * p * t_inv).1;

    assert_eq!(expected, actual);

    // panic!("done");
}

#[test]
fn scalar_division_associativity() {
    let b = Bivector::new(2., 3., 5., 7., 11., 13., 17., 19., 23., 29.);
    assert_eq!(b * (b * 2.), b * b * 2.);
    assert_eq!(b * 0.5, b / 2.);
    assert_eq!((b * (b / 2.)).2, (b * b / 2.).2);
    assert_eq!((b * (b / 2.)).4, (b * b / 2.).4);
    assert_eq!((b * (b / 2.)).0, (b * b / 2.).0);
}

#[test]
fn sandwich_point() {
    let p = point(2., 0., 0.);
    let s = point(1., 0., 0.)
        .wedge(point(-1., 0., 0.))
        .wedge(point(0., 1., 0.))
        .wedge(point(0., 0., 1.));

    let s = s / s.dot(s).abs().sqrt();

    let p_ = s.dual().sandwich(p.dual()).dual();

    dbg!(p, p_, s.dual());

    // panic!("done");
}
