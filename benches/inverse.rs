use clifford::pga3::*;
use criterion::{criterion_group, criterion_main, Criterion};

criterion_main!(inverse);

criterion_group!(inverse, vector, unit_vector);
// criterion_group!(denominator, vector, bivector, motor, flector, multivector);

pub fn vector(crit: &mut Criterion) {
    let a = Motor {
        s: 1.,
        xy: 1.,
        yz: 1.,
        xz: 1.,
        xw: 1.,
        yw: 1.,
        zw: 1.,
        xyzw: 1.,
    };

    let b = Vector {
        x: 2.,
        y: 3.,
        z: 5.,
        w: 7.,
    };

    let mut c = Vector::default();

    crit.bench_function("motor", |bench| {
        bench.iter(|| {
            c = a.sandwich(b);
        })
    });
}

pub fn unit_vector(crit: &mut Criterion) {
    let a = Motor {
        s: 1.,
        xy: 1.,
        yz: 1.,
        xz: 1.,
        xw: 1.,
        yw: 1.,
        zw: 1.,
        xyzw: 1.,
    }
    .unit();

    let b = Vector {
        x: 2.,
        y: 3.,
        z: 5.,
        w: 7.,
    };

    let mut c = Vector::default();

    crit.bench_function("motor unit", |bench| {
        bench.iter(|| {
            c = a.sandwich(b);
        })
    });
}
