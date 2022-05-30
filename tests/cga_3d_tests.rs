// #[cfg(feature = "cga_3d")]
use clifford::cga_3d::*;

/// All references to Geometric Algebra for Computer Science

#[test]
fn bases() {
    let e_pos = Vector::new(0., 0., 0., 1., 0.);
    let e_neg = Vector::new(0., 0., 0., 0., 1.);

    // Table 13.1
    assert_eq!(1., e_pos.dot(e_pos));
    assert_eq!(-1., e_neg.dot(e_neg));
    assert_eq!(0., N.dot(N));
    assert_eq!(0., N_BAR.dot(N_BAR));
    assert_eq!(-1., N_BAR.dot(N));
    assert_eq!(-1., N.dot(N_BAR));

    assert_eq!(e_neg - e_pos, N_BAR); // 13.5
    assert_eq!((e_neg + e_pos) / 2., N); // 13.5
    assert_eq!(e_pos, N - 0.5 * N_BAR); // 13.6
    assert_eq!(e_neg, N + 0.5 * N_BAR); // 13.6
}

#[test]
fn point_distance() {
    let a = N;
    let b = point(3., 4., 0.);

    assert_eq!(-12.5, a.dot(b)); // 13.4
    assert_eq!(0., b.dot(b));
}

#[test]
fn normalized_point_aka_unit_point() {
    let p = point(1., 2., 3.);
    assert_eq!(1.0, -N_BAR.dot(p)); // Section 13.1.2
    assert_eq!(1.0, -N_BAR.dot(N)); // Section 13.1.2
}

#[test]
fn dual() {
    let e1 = Vector::new(1., 0., 0., 0., 0.);
    let e2 = Vector::new(0., 1., 0., 0., 0.);
    let e3 = Vector::new(0., 0., 1., 0., 0.);
    let e4 = Vector::new(0., 0., 0., 1., 0.);
    let e5 = Vector::new(0., 0., 0., 0., 1.);

    let e124 = e1.wedge(e2).wedge(e4);

    assert_eq!(e124.right_comp(), -e3.wedge(e5));
}

#[allow(dead_code)]
fn invert1(v: Vector) -> Vector {
    const E4: Vector = Vector::new(0., 0., 0., -1., 0.);
    -E4.geo(v).geo(E4).1
}

#[allow(dead_code)]
fn invert2(v: Bivector) -> Bivector {
    const E4: Vector = Vector::new(0., 0., 0., -1., 0.);
    -E4.geo(v).geo(E4).2
}

#[allow(dead_code)]
fn invert3(v: Trivector) -> Trivector {
    const E4: Vector = Vector::new(0., 0., 0., -1., 0.);
    -E4.geo(v).geo(E4).3
}

#[allow(dead_code)]
fn invert4(v: Quadvector) -> Quadvector {
    const E4: Vector = Vector::new(0., 0., 0., -1., 0.);
    -E4.geo(v).geo(E4).4
}

pub trait Invert {
    fn inv(self) -> Self;
}

impl Invert for Vector {
    fn inv(self) -> Vector {
        const E4: Vector = Vector::new(0., 0., 0., -1., 0.);
        -E4.geo(self).geo(E4).1
    }
}

impl Invert for Bivector {
    fn inv(self) -> Bivector {
        const E4: Vector = Vector::new(0., 0., 0., -1., 0.);
        -E4.geo(self).geo(E4).2
    }
}

impl Invert for Trivector {
    fn inv(self) -> Trivector {
        const E4: Vector = Vector::new(0., 0., 0., -1., 0.);
        -E4.geo(self).geo(E4).3
    }
}

impl Invert for Quadvector {
    fn inv(self) -> Quadvector {
        const E4: Vector = Vector::new(0., 0., 0., -1., 0.);
        -E4.geo(self).geo(E4).4
    }
}

#[test]
fn dual_plane_test() {
    let plane = point(0., 0., 1.)
        .wedge(point(1., 0., 1.))
        .wedge(point(0., 1., 1.))
        .wedge(N_BAR);

    let plane_dual = plane.inv().right_comp(); // why is this inverse here?

    let expected = Vector::new(0., 0., 1., 0., 0.) - 1. * N_BAR;

    assert_eq!(plane_dual, expected);
    assert_eq!(0., N_BAR.dot(plane_dual));
    assert_eq!(1., plane_dual.dot(plane_dual));

    // panic!("{:#?}", plane_dual);
}

#[test]
fn dual_real_sphere() {
    // centred at (0,0,1), r=1
    let sphere = N
        .wedge(point(0., 0., 2.))
        .wedge(point(1., 0., 1.))
        .wedge(point(0., 1., 1.));

    let sphere_dual = sphere.right_comp() * 0.5;

    let expected = Vector::new(0., 0., 1., 0., 0.) - 0.5 * 1. * N_BAR;

    assert_eq!(sphere_dual, expected);

    // panic!("{:#?}", sphere_dual);
}

pub type Translator = Multivector<f64, Zero, Bivector, Zero, Zero, Zero>;

pub fn translator(x: f64, y: f64, z: f64) -> (Translator, Translator) {
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

    assert_eq!((t * t_inv).0, 1.0);
    assert_eq!((t * t_inv).2, Default::default());
    assert_eq!((t * t_inv).4, Default::default());

    let expected = point(3., 5., 8.);
    let actual = (t * p * t_inv).1;

    assert_eq!(expected, actual);

    // panic!("done");
}

#[test]
fn bivector_test() {
    let b = Bivector::new(2., 3., 5., 7., 11., 13., 17., 19., 23., 29.);
    dbg!(b);
    let b_ = Bivector::new(2., 3., 5., 7., 11., 13., 17., 19., 23., 29.);
    // let b = Bivector::new(2., 3., 0., 0., 0., 0., 0., 0., 0., 0.);
    // let b_ = Bivector::new(2., 3., 0., 0., 0., 0., 0., 0., 0., 0.));

    panic!("{:#?}", b * b_);
    panic!();
}

// #[test]
// fn scalar_division_associativity() {
//     let b = Bivector::new(2., 3., 5., 7., 11., 13., 17., 19., 23., 29.);
//     assert_eq!(b * (b * 2.), b * b * 2.);
//     assert_eq!(b * 0.5, b / 2.);
//     assert_eq!((b * (b / 2.)).2, (b * b / 2.).2);
//     assert_eq!((b * (b / 2.)).4, (b * b / 2.).4);
//     assert_eq!((b * (b / 2.)).0, (b * b / 2.).0);
// }
