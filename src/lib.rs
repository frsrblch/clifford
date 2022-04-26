#![feature(
    const_trait_impl,
    const_fn_floating_point_arithmetic,
    const_default_impls,
    const_convert
)]

// TODO consider grade enum model with dynamic multivectors (e.g., BTreeMap<Grade, f64>?)
// TODO create proc_macro crate
// TODO grade sub, heterogeneous add/sub,reverses, complements, antiproducts

// /// Cl(4,2,0) - 2D conics??
// #[cfg(test)]
// mod testing {
//     use super::*;
//
//     const N_X: Vector = Vector::new(0., 0., 1., 0., 1., 0.);
//     const N_Y: Vector = Vector::new(0., 0., 0., 1., 0., 1.);
//
//     const N_X_BAR: Vector = Vector::new(0., 0., 1., 0., -1., 0.);
//     const N_Y_BAR: Vector = Vector::new(0., 0., 0., 1., 0., -1.);
//
//     #[test]
//     fn conic_consts() {
//         assert_eq!(0., (N_X.dot(N_X)));
//         assert_eq!(0., (N_Y.dot(N_Y)));
//         assert_eq!(0., (N_X_BAR.dot(N_X_BAR)));
//         assert_eq!(0., (N_Y_BAR.dot(N_Y_BAR)));
//         assert_eq!(2., (N_Y.dot(N_Y_BAR)));
//     }
//
//     fn point(x: f64, y: f64) -> Vector {
//         let x2 = x * x;
//         let y2 = y * y;
//         let x = Vector::new(x, y, 0., 0., 0., 0.);
//
//         2.0 * x + x2 * N_X + y2 * N_Y - N_X_BAR - N_Y_BAR
//     }
//
//     #[test]
//     fn wedge_points() {
//         let o = point(0., 0.);
//         let x = point(1., 0.);
//         let y = point(0., 1.);
//
//         dbg!(o.wedge(x));
//         dbg!(o.wedge(y));
//         dbg!(o.wedge(x).wedge(y));
//         panic!();
//     }
// }

// /// 3D CGA
// /// Point at the origin
// const N_BAR: Vector = Vector::new(0., 0., 0., 1., -1.);
//
// /// Point through infinity
// const N: Vector = Vector::new(0., 0., 0., 1., 1.);
//
// /// GA for Physicists, Chapter 10
// fn point(x: f64, y: f64, z: f64) -> Vector {
//     let x2 = x * x + y * y + z * z;
//     let x = Vector::new(x, y, z, 0., 0.);
//     x2 * N + 2.0 * x - N_BAR
// }
//
// /// Check whether point passes through infinity
// fn is_flat<T, W>(value: T) -> bool
// where
//     Vector: Wedge<T, Output = W>,
//     W: PartialEq + From<Zero>,
// {
//     N.wedge(value) == W::from(Zero)
// }
//
// #[test]
// fn null_squares() {
//     assert_eq!(N_BAR * N_BAR, Even::default());
//
//     assert_eq!(N * N, Even::default());
//
//     assert_eq!(N_BAR.dot(N), 2.);
//
//     assert_eq!(
//         N_BAR.wedge(N),
//         Bivector::new(0., 0., 0., 0., 0., 0., 0., 0., 0., 2.0)
//     );
// }
//
// #[test]
// fn point_pair() {
//     let pt = point(0., 0., 0.);
//     let b = point(1., 0., 0.);
//     let c = point(0., 1., 0.);
//     let d = point(0., 0., 1.);
//
//     assert_eq!(pt.dot(b), -2.);
//     assert!(!is_flat(pt));
//
//     let pair = pt.wedge(b);
//     assert!(!is_flat(pair));
//
//     let line = pair.wedge(N_BAR);
//     assert!(is_flat(line));
//
//     let plane = line.wedge(c);
//     assert!(is_flat(plane));
//
//     let circle = pair.wedge(c);
//     assert!(!is_flat(circle));
//
//     let sphere = circle.wedge(d);
//     assert!(!is_flat(sphere));
// }

pub mod pga_3d {
    proc_macros::clifford!(3, 1, 0);

    #[test]
    fn vec_mul() {
        let a = Vector::new(2., 3., 4., 1.);
        let b = Vector::new(3., 5., 7., 1.);
        let c = Vector::new(0., 0., 0., 1.);

        let line = a.wedge(b);
        let plane = line.wedge(c);

        dbg!(line, plane);
        // panic!("done");
    }
}
