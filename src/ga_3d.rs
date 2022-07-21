proc_macros::clifford!(3, 0, 0, [x, y, z]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_autodif_test() {
        let x = Vector::new(1f32, 2., 3.);
        let v = Vector::new(-5f32, 7., 11.);

        let ux = x / x.dot(x).sqrt();
        let uv = v / v.dot(v).sqrt();

        let p = ux * uv;
        dbg!(p);

        // panic!("{:?}", p);
    }

    #[test]
    fn accel_derivative() {
        let a = |x: Vector<f64>| -x / x.dot(x).powf(1.5);
        let x_0 = Vector::new(0f64, 0., 2.);

        let norm = |v: Vector<f64>| v.dot(v).sqrt();

        let da = |x: Vector<f64>| {
            let inv_det = x.dot(x).powf(-2.5);
            let Vector { x, y, z } = x;
            let (x2, y2, z2) = (x * x, y * y, z * z);
            let da = Vector::new(
                -2. * x2 + y2 + z2 - 3. * x * (y + z),
                x2 - 2. * y2 + z2 - 3. * y * (x + z),
                x2 + y2 - 2. * z2 - 3. * z * (x + y),
            ) * inv_det;
            move |h: Vector<f64>| Vector::new(da.x * h.x, da.y * h.y, da.z * h.z) / norm(h)
        };

        let dt = 0.05f64;
        let dt2 = dt * dt;
        let mut x = x_0;
        let mut v = Vector::new(0f64, 0.01, 0.02);

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
