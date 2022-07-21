// #[cfg(feature = "cga_3d")]
use clifford::cga_3d::*;

/// All references to Geometric Algebra for Computer Science

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

pub trait Inverse {
    fn inv(self) -> Self;
}

// TODO generic implementation using GradeFilter<T>
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

impl<T, G0, G1, G2, G3, G4, G5> GradeFilter<Multivector<G0, G1, G2, G3, G4, G5>> for Vector<T> {
    type Output = G1;

    fn filter(value: Multivector<G0, G1, G2, G3, G4, G5>) -> Self::Output {
        value.1
    }
}

impl<G0, G1, G2, G3, G4, G5> GradeFilter<Multivector<G0, G1, G2, G3, G4, G5>> for f32 {
    type Output = G0;

    fn filter(value: Multivector<G0, G1, G2, G3, G4, G5>) -> Self::Output {
        value.0
    }
}

impl<G0, G1, G2, G3, G4, G5> GradeFilter<Multivector<G0, G1, G2, G3, G4, G5>> for f64 {
    type Output = G0;

    fn filter(value: Multivector<G0, G1, G2, G3, G4, G5>) -> Self::Output {
        value.0
    }
}

impl<T, G0, G1, G2, G3, G4, G5> GradeFilter<Multivector<G0, G1, G2, G3, G4, G5>> for Bivector<T> {
    type Output = G2;

    fn filter(value: Multivector<G0, G1, G2, G3, G4, G5>) -> Self::Output {
        value.2
    }
}

impl<T, G0, G1, G2, G3, G4, G5> GradeFilter<Multivector<G0, G1, G2, G3, G4, G5>> for Trivector<T> {
    type Output = G3;

    fn filter(value: Multivector<G0, G1, G2, G3, G4, G5>) -> Self::Output {
        value.3
    }
}

impl<T, G0, G1, G2, G3, G4, G5> GradeFilter<Multivector<G0, G1, G2, G3, G4, G5>> for Quadvector<T> {
    type Output = G4;

    fn filter(value: Multivector<G0, G1, G2, G3, G4, G5>) -> Self::Output {
        value.4
    }
}

impl<T, G0, G1, G2, G3, G4, G5> GradeFilter<Multivector<G0, G1, G2, G3, G4, G5>>
    for Pentavector<T>
{
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

    let s = s / s.dot(s).abs().sqrt();

    let p_ = s.dual().sandwich(p.dual()).dual();

    dbg!(p, p_, s.dual());

    // panic!("done");
}
