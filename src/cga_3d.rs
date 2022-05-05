proc_macros::clifford!(4, 1, 0);

/// Point at the origin
pub const N: Vector = Vector::new(0., 0., 0., 0.5, 0.5);

/// Point through infinity
pub const N_BAR: Vector = Vector::new(0., 0., 0., -1., 1.);

pub const fn point(x: f64, y: f64, z: f64) -> Vector {
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

trait IsFlat {
    fn is_flat(self) -> bool;
}

impl<T, U> IsFlat for T
where
    T: crate::Wedge<Vector, Output = U>,
    U: Default + PartialEq,
{
    fn is_flat(self) -> bool {
        self.wedge(N_BAR) == U::default()
    }
}

#[test]
fn flat_test() {
    let o = point(0., 0., 0.);
    let a = point(1., 0., 0.);
    let b = point(0., 1., 0.);
    let c = point(0., 0., 1.);

    let points: Bivector = o.wedge(a);
    let line: Trivector = points.wedge(N_BAR);
    let circle: Trivector = points.wedge(b);
    let plane: Quadvector = line.wedge(c);
    let sphere: Quadvector = circle.wedge(c);

    assert!(!o.is_flat());
    assert!(!points.is_flat());
    assert!(!circle.is_flat());
    assert!(!sphere.is_flat());
    assert!(line.is_flat());
    assert!(plane.is_flat());
}
