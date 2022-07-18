proc_macros::clifford!(3, 0, 0);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_autodif_test() {
        let x = Vector::new(1., 2., 3.);
        let v = Vector::new(-5., 7., 11.);

        let ux = x / x.dot(x).sqrt();
        let uv = v / v.dot(v).sqrt();

        let p = ux * uv;
        dbg!(p);

        // panic!("{:?}", p);
    }

    #[test]
    fn accel_derivative() {
        let a = |x: Vector| -x / x.dot(x).powf(1.5);
        let x_0 = Vector::new(0., 0., 2.);

        let norm = |v: Vector| v.dot(v).sqrt();

        let da = |x: Vector| {
            let inv_det = x.dot(x).powf(-2.5);
            let Vector {
                e1: x,
                e2: y,
                e3: z,
            } = x;
            let (x2, y2, z2) = (x * x, y * y, z * z);
            let da = Vector::new(
                -2. * x2 + y2 + z2 - 3. * x * (y + z),
                x2 - 2. * y2 + z2 - 3. * y * (x + z),
                x2 + y2 - 2. * z2 - 3. * z * (x + y),
            ) * inv_det;
            move |h: Vector| Vector::new(da.e1 * h.e1, da.e2 * h.e2, da.e3 * h.e3) / norm(h)
        };

        let dt = 0.05;
        let dt2 = dt * dt;
        let mut x = x_0;
        let mut v = Vector::new(0., 0.01, 0.02);

        dbg!(x, v);
        for _ in 0..20 {
            let a = a(x) + 0.5 * da(x)(v * dt);

            x = x + dt * v + 0.5 * dt2 * a;
            v = v + dt * a;

            dbg!(x, v);
        }

        // panic!("done");
    }

    #[test]
    fn g3_contains_dual() {
        let vector = Vector::new(1., 2., 3.);
        let expected = Bivector::new(3., -2., 1.);
        assert_eq!(expected, vector.dual());
    }

    #[test]
    fn dual_is_symmetrical() {
        let vector = Vector::new(1., 2., 3.);
        assert_eq!(vector, vector.dual().dual());
    }
}
