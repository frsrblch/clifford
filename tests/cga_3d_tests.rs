use clifford::cga_3d::*;

#[test]
fn is_flat_tests() {
    #[track_caller]
    fn assert_null_non_zero<T, N>(t: T)
    where
        T: Zero + Geo<T, Output = N> + Copy,
        N: Zero,
    {
        assert!(!t.is_zero());
        assert!((t.geo(t)).is_zero());
    }

    #[track_caller]
    fn assert_flat<T, N>(t: T)
    where
        T: Copy + Zero + Geo<T, Output = N> + IsFlat + Copy,
        N: Zero,
    {
        assert!(!t.is_zero());
        assert!(t.is_flat());
    }

    #[track_caller]
    fn assert_round<T, N>(t: T)
    where
        T: Copy + Zero + Geo<T, Output = N> + IsFlat + Copy,
        N: Zero,
    {
        assert!(!t.is_zero());
        assert!(!t.is_flat());
    }

    let o = origin::<f64>();
    let inf = infinity::<f64>();
    let a = point(1., 0., 0.);
    let b = point(0., 1., 0.);
    let c = point(0., 0., 1.);

    assert_null_non_zero(a);
    assert_null_non_zero(o);
    assert_null_non_zero(inf);

    let flat_point = a ^ inf;
    assert_flat(flat_point);

    let point_pair = a ^ o;
    assert_round(point_pair);

    let line = a ^ b ^ inf;
    assert_flat(line);

    let circle = o ^ a ^ b;
    assert_round(circle);

    let plane = a ^ b ^ c ^ inf;
    assert_flat(plane);

    let sphere = o ^ a ^ b ^ c;
    assert_round(sphere);

    let volume = plane ^ o;
    assert!(!volume.is_zero());

    let volume = sphere ^ inf;
    assert!(!volume.is_zero());
}

fn invert<T, U>(t: T) -> T
where
    T: FloatType + std::ops::Neg<Output = T>,
    Vector<T::Float>: Geo<T, Output = U>,
    T: GradeProduct<U, Vector<T::Float>, Output = T>,
{
    let e = Vector::<T::Float> {
        e: T::Float::one(),
        ..zero()
    };
    -T::product(e.geo(t), e)
}

#[test]
fn invert_line_eq_circle_through_origin() {
    let a = point(0., 1., 0.);
    let b = point(0., 0., 1.);
    let o = origin::<f64>();
    let line = a ^ b ^ infinity::<f64>();

    assert!(!(line ^ o).is_zero());

    let circle = invert(line);
    assert!(!circle.is_zero());
    assert!((circle ^ o).is_zero());
}

#[test]
fn point_reflection() {
    let pt = point(2., 3., 5.);
    let dual = pt.dual();

    dbg!(pt * pt.rev(), dual * dual.rev());
    assert!(pt.geo(pt).is_zero());
    assert!(dual.geo(dual).is_zero());
}

#[test]
fn vector_test() {
    let x = Vector::new(2., 3., 5., 0., 0.);
    let x2 = 2. * 2. + 3. * 3. + 5. * 5.;
    let expected = x + 0.5 * x2 * infinity::<f64>() + origin::<f64>();

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
    let line: Trivector<f64> = points.wedge(infinity::<f64>());
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
    let o = origin::<f64>();
    let inf = infinity::<f64>();

    let p = Vector::new(0., 0., 0., 1., 0.);
    let n = Vector::new(0., 0., 0., 0., 1.);

    assert_eq!(1., p.dot(p));
    assert_eq!(-1., n.dot(n));
    assert_eq!(0., o.dot(o));
    assert_eq!(0., inf.dot(inf));
    assert_eq!(-1., inf.dot(o));
    assert_eq!(-1., o.dot(inf));

    assert_eq!(n - p, inf);
    assert_eq!((n + p) / 2., o);
    assert_eq!(p, o - 0.5 * inf);
    assert_eq!(n, o + 0.5 * inf);
}

#[test]
fn point_distance() {
    let a = origin::<f64>();
    let b = point(3., 4., 0.);

    assert_eq!(-12.5, a.dot(b));
    assert_eq!(0., b.dot(b));
}

#[test]
fn normalized_point_aka_unit_point() {
    let o = origin::<f64>();
    let inf = infinity::<f64>();
    let p = point(1., 2., 3.);

    assert_eq!(1., -inf.dot(p));
    assert_eq!(1., -inf.dot(o));
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
    let inf = infinity::<f64>();

    let plane = point(0f64, 0., 1.)
        .wedge(point(1., 0., 1.))
        .wedge(point(0., 1., 1.))
        .wedge(inf);

    let plane_dual = plane.dual();

    let expected = Vector::new(0., 0., 1., 0., 0.) - 1. * inf;

    assert_eq!(invert(plane_dual), expected);
    assert_eq!(2., inf.dot(plane_dual));
    assert_eq!(1., plane_dual.dot(plane_dual));
}

#[test]
fn dual_real_sphere() {
    let o = origin::<f64>();
    let inf = infinity::<f64>();

    // centred at (0,0,1), r=1
    let sphere = o
        .wedge(point(0., 0., 2.))
        .wedge(point(1., 0., 1.))
        .wedge(point(0., 1., 1.));

    let sphere_dual = sphere.dual() * 0.5;

    let expected = Vector::new(0., 0., 1., 0., 0.) - 0.5 * inf;

    assert_eq!(sphere_dual, expected);
}

#[test]
fn point_inversion() {
    let o = origin::<f64>();
    let inf = infinity::<f64>();
    let p = point(2., 3., 5.);

    dbg!(p.null_bases());

    // assert is unit point
    assert_eq!(-1., p.dot(inf));

    // reflect origin through point

    let p_ = Vector::product(p.geo(o), invert(p)) / -38.;
    dbg!(p, p_);
    assert_eq!(-1., p_.dot(inf));
    dbg!(p_.null_bases());

    // returns the origin reflected across a sphere with zero radius, i.e., the point itself
    // panic!("\n{:#?}\n{:#?}", p, p_);
}

#[test]
fn null_bases() {
    let o = origin::<f64>();
    let inf = infinity::<f64>();
    assert_eq!((1., 0.), o.null_bases());
    assert_eq!((0., 1.), inf.null_bases());
    assert_eq!((1., 1.), (o + inf).null_bases());
    assert_eq!((2., 3.), (2. * o + 3. * inf).null_bases());
    assert_eq!((1., 0.5 * (4. + 9. + 25.)), point(2., 3., 5.).null_bases());
}

#[test]
fn vector_type() {
    let o = origin::<f64>();
    let inf = infinity::<f64>();
    let origin_plane = Vector::new(1f64, 0., 0., 0., 0.);
    let plane = Vector::new(1f64, 0., 0., 0., 0.) + inf;
    let sphere_r7 = point(2f64, 3., 5.) - (0.5 * 7. * 7.) * inf;
    let sphere_rneg7 = point(2f64, 3., 5.) + (0.5 * 7. * 7.) * inf;

    assert_eq!(VectorType::Infinity, inf.vector_type());
    assert_eq!(VectorType::Infinity, (inf * 2.).vector_type());
    assert_eq!(VectorType::Infinity, (inf * -2.).vector_type());

    assert_eq!(VectorType::DualPlane, origin_plane.vector_type());
    assert_eq!(VectorType::DualPlane, (origin_plane * 2.).vector_type());
    assert_eq!(VectorType::DualPlane, (origin_plane * -2.).vector_type());

    assert_eq!(VectorType::DualPlane, plane.vector_type());
    assert_eq!(VectorType::DualPlane, (plane * 2.).vector_type());
    assert_eq!(VectorType::DualPlane, (plane * -2.).vector_type());

    assert_eq!(VectorType::DualSphere(0.), o.vector_type());
    assert_eq!(VectorType::DualSphere(0.), (o * 2.).vector_type());
    assert_eq!(VectorType::DualSphere(0.), (o * -2.).vector_type());

    assert_eq!(VectorType::DualSphere(0.), point(1., 2., 3.).vector_type());
    assert_eq!(
        VectorType::DualSphere(0.),
        (point(1f64, 2., 3.) * 2.).vector_type()
    );
    assert_eq!(
        VectorType::DualSphere(0.),
        (point(1f64, 2., 3.) * -2.).vector_type()
    );

    assert_eq!(VectorType::DualSphere(7.), sphere_r7.vector_type());
    assert_eq!(VectorType::DualSphere(7.), (sphere_r7 * 2.).vector_type());
    assert_eq!(VectorType::DualSphere(7.), (sphere_r7 * -2.).vector_type());

    assert_eq!(VectorType::DualSphere(-7.), sphere_rneg7.vector_type());
    assert_eq!(
        VectorType::DualSphere(-7.),
        (sphere_rneg7 * 2.).vector_type()
    );
    assert_eq!(
        VectorType::DualSphere(-7.),
        (sphere_rneg7 * -2.).vector_type()
    );
}
