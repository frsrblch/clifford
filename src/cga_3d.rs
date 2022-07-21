proc_macros::clifford!(4, 1, 0, [x, y, z, e, E]);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_test() {
        let x = Vector::new(2., 3., 5., 0., 0.);
        let x2 = 2. * 2. + 3. * 3. + 5. * 5.;
        let expected = x + 0.5 * x2 * N_BAR + N;

        let actual = point(2., 3., 5.);

        assert_eq!(expected, actual);
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

        assert_eq!(e124.dual(), -e3.wedge(e5));
    }

    #[test]
    fn dual_plane_test() {
        let plane = point(0., 0., 1.)
            .wedge(point(1., 0., 1.))
            .wedge(point(0., 1., 1.))
            .wedge(N_BAR);

        fn inv(x: Vector<f64>) -> Vector<f64> {
            const E4: Vector<f64> = Vector::new(0., 0., 0., -1., 0.);
            -E4.geo(x).geo(E4).1
        }

        let plane_dual = plane.dual();
        let plane_dual_inv = plane_dual / plane_dual.scalar_prod(plane_dual);

        let expected = Vector::new(0., 0., 1., 0., 0.) - 1. * N_BAR;

        dbg!(plane_dual * plane_dual_inv, plane_dual * plane_dual);

        assert_eq!(inv(plane_dual), expected);
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

        let sphere_dual = sphere.dual() * 0.5;

        let expected = Vector::new(0., 0., 1., 0., 0.) - 0.5 * 1. * N_BAR;

        assert_eq!(sphere_dual, expected);

        // panic!("{:#?}", sphere_dual);
    }
}
