use clifford::{pga_point, Vector, Wedge};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

criterion_main! { mul }
criterion_group! { mul, mul_mv, mul_ty }

fn mul_mv(crit: &mut Criterion) {
    let a = pga_point(0., 0., 0.);
    let b = pga_point(1., 1., 1.);
    let c = pga_point(2., 3., 5.);

    let ab = a.wedge(&b);
    let ac = a.wedge(&c);

    assert_eq(8, ab.wedge(&ac).len());

    crit.bench_function("mul_mv", |b| {
        b.iter(|| {
            black_box(&ab * &ac);
        })
    });
}

fn mul_ty(crit: &mut Criterion) {
    let a = Vector::new(0., 0., 0., 1.);
    let b = Vector::new(1., 1., 1., 1.);
    let c = Vector::new(2., 3., 5., 1.);

    let ab = a.wedge(b);
    let ac = a.wedge(c);

    let abc = (ab * ac);

    crit.bench_function("mul_ty", |b| {
        b.iter(|| {
            black_box(ab * ac);
        })
    });
}
