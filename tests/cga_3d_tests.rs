// #[cfg(feature = "cga_3d")]
use clifford::cga_3d::*;
use clifford::Geometric;

/// All references to Geometric Algebra for Computer Science

#[test]
fn bases() {
    let e_pos = Vector::new(0., 0., 0., 1., 0.);
    let e_neg = Vector::new(0., 0., 0., 0., 1.);

    // Table 13.1
    assert_eq!(1., e_pos.dot(e_pos).value);
    assert_eq!(-1., e_neg.dot(e_neg).value);
    assert_eq!(0., N.dot(N).value);
    assert_eq!(0., N_BAR.dot(N_BAR).value);
    assert_eq!(-1., N_BAR.dot(N).value);
    assert_eq!(-1., N.dot(N_BAR).value);

    assert_eq!(e_neg - e_pos, N_BAR); // 13.5
    assert_eq!((e_neg + e_pos) / 2., N); // 13.5
    assert_eq!(e_pos, N - 0.5 * N_BAR); // 13.6
    assert_eq!(e_neg, N + 0.5 * N_BAR); // 13.6
}

#[test]
fn point_distance() {
    let a = N;
    let b = point(3., 4., 0.);

    assert_eq!(-12.5, a.dot(b).value); // 13.4
    assert_eq!(0., b.dot(b).value);
}

#[test]
fn normalized_point_aka_unit_point() {
    let p = point(1., 2., 3.);
    assert_eq!(1.0, -N_BAR.dot(p).value); // Section 13.1.2
    assert_eq!(1.0, -N_BAR.dot(N).value); // Section 13.1.2
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

pub trait Inverse {
    fn inv(self) -> Self;
}

impl Inverse for Vector {
    fn inv(self) -> Vector {
        const E4: Vector = Vector::new(0., 0., 0., -1., 0.);
        -E4.geo(self).geo(E4).1
    }
}

impl Inverse for Bivector {
    fn inv(self) -> Bivector {
        const E4: Vector = Vector::new(0., 0., 0., -1., 0.);
        -E4.geo(self).geo(E4).2
    }
}

impl Inverse for Trivector {
    fn inv(self) -> Trivector {
        const E4: Vector = Vector::new(0., 0., 0., -1., 0.);
        -E4.geo(self).geo(E4).3
    }
}

impl Inverse for Quadvector {
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
    assert_eq!(Scalar::default(), N_BAR.dot(plane_dual));
    assert_eq!(Scalar::new(1.), plane_dual.dot(plane_dual));

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

pub type Translator = Multivector<Scalar, Zero, Bivector, Zero, Zero, Zero>;

pub fn translator(x: f64, y: f64, z: f64) -> (Translator, Translator) {
    let t = Vector::new(x, y, z, 0., 0.);
    let b = 0.5 * t * N_BAR;

    // this works because N_BAR * N_BAR = 0 for any t
    // so the reversed bivector sign is enough to make them cancel out
    (Scalar::new(1.) - b, Scalar::new(1.) + b)
}

#[test]
#[allow(non_snake_case)]
fn translation_test() {
    let (t, t_inv) = translator(2., 3., 5.);
    let p = point(1., 2., 3.);

    assert_eq!((t * t_inv).0, Scalar::new(1.));
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

pub trait GradeFilter<T> {
    type Output;
    fn filter(value: T) -> Self::Output;
}

impl<T> GradeFilter<T> for Zero {
    type Output = Zero;
    #[inline]
    #[allow(unused_variables)]
    fn filter(value: T) -> Self::Output {
        Zero
    }
}

impl<G0, G1, G2, G3, G4, G5> GradeFilter<Multivector<G0, G1, G2, G3, G4, G5>> for Vector {
    type Output = G1;

    fn filter(value: Multivector<G0, G1, G2, G3, G4, G5>) -> Self::Output {
        value.1
    }
}

impl<G0, G1, G2, G3, G4, G5> GradeFilter<Multivector<G0, G1, G2, G3, G4, G5>> for f64 {
    type Output = G0;

    fn filter(value: Multivector<G0, G1, G2, G3, G4, G5>) -> Self::Output {
        value.0
    }
}

impl<G0, G1, G2, G3, G4, G5> GradeFilter<Multivector<G0, G1, G2, G3, G4, G5>> for Bivector {
    type Output = G2;

    fn filter(value: Multivector<G0, G1, G2, G3, G4, G5>) -> Self::Output {
        value.2
    }
}

impl<G0, G1, G2, G3, G4, G5> GradeFilter<Multivector<G0, G1, G2, G3, G4, G5>> for Trivector {
    type Output = G3;

    fn filter(value: Multivector<G0, G1, G2, G3, G4, G5>) -> Self::Output {
        value.3
    }
}

impl<G0, G1, G2, G3, G4, G5> GradeFilter<Multivector<G0, G1, G2, G3, G4, G5>> for Quadvector {
    type Output = G4;

    fn filter(value: Multivector<G0, G1, G2, G3, G4, G5>) -> Self::Output {
        value.4
    }
}

impl<G0, G1, G2, G3, G4, G5> GradeFilter<Multivector<G0, G1, G2, G3, G4, G5>> for Pentavector {
    type Output = G5;

    fn filter(value: Multivector<G0, G1, G2, G3, G4, G5>) -> Self::Output {
        value.5
    }
}

impl<G0, G1, G2, G3, G4, G5, Rhs> GradeFilter<Rhs> for Multivector<G0, G1, G2, G3, G4, G5>
where
    Rhs: Copy,
    G0: GradeFilter<Rhs>,
    G1: GradeFilter<Rhs>,
    G2: GradeFilter<Rhs>,
    G3: GradeFilter<Rhs>,
    G4: GradeFilter<Rhs>,
    G5: GradeFilter<Rhs>,
{
    type Output =
        Multivector<G0::Output, G1::Output, G2::Output, G3::Output, G4::Output, G5::Output>;

    fn filter(value: Rhs) -> Self::Output {
        Multivector(
            G0::filter(value),
            G1::filter(value),
            G2::filter(value),
            G3::filter(value),
            G4::filter(value),
            G5::filter(value),
        )
    }
}

pub trait SandwichProduct<Inner> {
    type Output;
    fn sandwich(self, inner: Inner) -> Self::Output;
}

impl<Inner, Outer, T0, T1> SandwichProduct<Inner> for Outer
where
    Outer: Geometric<Inner, Output = T0> + Inverse + Copy,
    T0: Geometric<Outer, Output = T1>,
    Inner: GradeFilter<T1>,
{
    type Output = <Inner as GradeFilter<T1>>::Output;

    fn sandwich(self, inner: Inner) -> Self::Output {
        let output = self.geo(inner).geo(self.inv());
        Inner::filter(output)
    }
}

#[test]
fn sandwich_point() {
    let p = point(2., 0., 0.);
    let s = point(1., 0., 0.)
        .wedge(point(-1., 0., 0.))
        .wedge(point(0., 1., 0.))
        .wedge(point(0., 0., 1.));

    let s = s / s.dot(s).value.abs().sqrt();

    let p_ = s.left_comp().sandwich(p.left_comp()).right_comp();

    dbg!(p, p_, s.left_comp());

    // panic!("done");
}
