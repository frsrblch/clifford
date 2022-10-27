use clifford::pga3::*;
use criterion::{criterion_group, criterion_main, Criterion};

criterion_main!(sandwich);
criterion_group!(
    sandwich,
    sandwich_non_closure_non_unit,
    sandwich_non_closure_unit,
    sandwich_closure_non_unit,
    sandwich_closure_unit
);

const N: usize = 64;

fn sandwich_non_closure_non_unit(crit: &mut Criterion) {
    let (sin, cos) = std::f64::consts::FRAC_PI_4.sin_cos();
    let motor = Motor::new(sin, cos, 0., 0., 0., 0., 1., 0.) * 2.;

    let mut vectors = vec![Vector::new(1., 2., 3., 4.); N];

    crit.bench_function("sandwich_non_closure_non_unit", |b| {
        b.iter(|| {
            for vector in &mut vectors {
                *vector = motor.sandwich(*vector);
            }
        })
    });
}

fn sandwich_non_closure_unit(crit: &mut Criterion) {
    let (sin, cos) = std::f64::consts::FRAC_PI_4.sin_cos();
    let motor = Motor::new(sin, cos, 0., 0., 0., 0., 0., 0.).unit();

    let mut vectors = vec![Vector::new(1., 2., 3., 4.); N];

    crit.bench_function("sandwich_non_closure_unit", |b| {
        b.iter(|| {
            for vector in &mut vectors {
                *vector = motor.sandwich(*vector);
            }
        })
    });
}

fn sandwich_closure_non_unit(crit: &mut Criterion) {
    let (sin, cos) = std::f64::consts::FRAC_PI_4.sin_cos();
    let motor = Motor::new(sin, cos, 0., 0., 0., 0., 0., 0.) * 2.;

    let mut vectors = vec![Vector::new(1., 2., 3., 4.); N];

    let sandwich = {
        let inv = motor.inv();
        move |v: Vector<f64>| {
            let intermediate = motor.geo(v);
            Vector::product(intermediate, inv)
        }
    };

    crit.bench_function("sandwich_closure_non_unit", |b| {
        b.iter(|| {
            for vector in &mut vectors {
                *vector = sandwich(*vector);
            }
        })
    });
}

fn sandwich_closure_unit(crit: &mut Criterion) {
    let (sin, cos) = std::f64::consts::FRAC_PI_4.sin_cos();
    let motor = Motor::new(sin, cos, 0., 0., 0., 0., 0., 0.).unit();

    let mut vectors = vec![Vector::new(1., 2., 3., 4.); N];

    let sandwich = {
        let inv = motor.inv();
        move |v: Vector<f64>| {
            let intermediate = motor.value().geo(v);
            Vector::product(intermediate, inv.value())
        }
    };

    crit.bench_function("sandwich_closure_unit", |b| {
        b.iter(|| {
            for vector in &mut vectors {
                *vector = sandwich(*vector);
            }
        })
    });
}
