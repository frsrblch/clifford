proc_macros::clifford!(4, 1, 0);

/// Point at the origin
pub const N: Vector<f64> = Vector::new(0., 0., 0., 0.5, 0.5);

/// Point through infinity
pub const N_BAR: Vector<f64> = Vector::new(0., 0., 0., -1., 1.);

pub fn translate(
    v: Vector<f64>,
    t: Multivector<f64, Zero, Bivector<f64>, Zero, Zero, Zero>,
) -> Vector<f64> {
    (t * v * t.rev()).1
}

pub const fn point(x: f64, y: f64, z: f64) -> Vector<f64> {
    let x2 = x * x + y * y + z * z;
    Vector::new(x, y, z, 0.5 - 0.5 * x2, 0.5 + 0.5 * x2)
}

#[test]
fn vector_test() {
    let x = Vector::new(2., 3., 5., 0., 0.);
    let x2 = 2. * 2. + 3. * 3. + 5. * 5.;
    let expected = x + 0.5 * x2 * N_BAR + N;

    let actual = point(2., 3., 5.);

    assert_eq!(expected, actual);
}

pub trait IsFlat {
    fn is_flat(&self) -> bool;
}

impl<T, U> IsFlat for T
where
    T: Wedge<Vector<f64>, Output = U> + Copy,
    U: Default + PartialEq,
{
    fn is_flat(&self) -> bool {
        self.wedge(N_BAR) == U::default()
    }
}

#[test]
fn flat_test() {
    let o = point(0., 0., 0.);
    let a = point(1., 0., 0.);
    let b = point(0., 1., 0.);
    let c = point(0., 0., 1.);

    let points: Bivector<f64> = o.wedge(a);
    let line: Trivector<f64> = points.wedge(N_BAR);
    let circle: Trivector<f64> = points.wedge(b);
    let plane: Quadvector<f64> = line.wedge(c);
    let sphere: Quadvector<f64> = circle.wedge(c);

    assert!(!o.is_flat());
    assert!(!points.is_flat());
    assert!(!circle.is_flat());
    assert!(!sphere.is_flat());
    assert!(line.is_flat());
    assert!(plane.is_flat());
}
