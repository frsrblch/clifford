use clifford::pga_3d::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use geo_traits::*;

criterion_main!(sandwich);
criterion_group!(
    sandwich,
    sandwich_non_closure_non_unit,
    sandwich_non_closure_unit,
    sandwich_closure_non_unit,
    sandwich_closure_unit
);

fn sandwich_non_closure_non_unit(crit: &mut Criterion) {
    let (sin, cos) = std::f32::consts::FRAC_PI_4.sin_cos();
    let motor = Motor::new(sin, cos, 0., 0., 0., 0., 1., 0.);

    let vector = Vector::new(1f32, 2., 3., 4.);
    let mut output = Vector::default();

    crit.bench_function("sandwich", |b| {
        b.iter(|| {
            output = black_box(motor) >> black_box(vector);
        })
    });
}

fn sandwich_non_closure_unit(crit: &mut Criterion) {
    let (sin, cos) = std::f32::consts::FRAC_PI_4.sin_cos();
    let motor = Motor::new(sin, cos, 0., 0., 0., 0., 0., 0.).unit();

    let vector = Vector::new(1f32, 2., 3., 4.);
    let mut output = Vector::default();

    crit.bench_function("sandwich_non_closure_unit", |b| {
        b.iter(|| {
            output = black_box(motor) >> black_box(vector);
        })
    });
}

fn sandwich_closure_non_unit(crit: &mut Criterion) {
    let (sin, cos) = std::f32::consts::FRAC_PI_4.sin_cos();
    let motor = Motor::new(sin, cos, 0., 0., 0., 0., 0., 0.);

    let vector = Vector::new(1f32, 2., 3., 4.);
    let mut output = Vector::default();

    let sandwich = {
        let inv = black_box(motor).inv();
        move |v: Vector<f32>| {
            let intermediate = black_box(motor).geo(v);
            Vector::product(intermediate, inv)
        }
    };

    crit.bench_function("sandwich_closure_non_unit", |b| {
        b.iter(|| {
            output = sandwich(black_box(vector));
        })
    });
}

fn sandwich_closure_unit(crit: &mut Criterion) {
    let (sin, cos) = std::f32::consts::FRAC_PI_4.sin_cos();
    let motor = Motor::new(sin, cos, 0., 0., 0., 0., 0., 0.).unit();

    let vector = Vector::new(1f32, 2., 3., 4.);
    let mut output = Vector::default();

    let sandwich = {
        let inv = black_box(motor).inv();
        move |v: Vector<f32>| {
            let intermediate = black_box(motor).value().geo(v);
            Vector::product(intermediate, inv.value())
        }
    };

    crit.bench_function("sandwich_closure_unit", |b| {
        b.iter(|| {
            output = sandwich(black_box(vector));
        })
    });
}
