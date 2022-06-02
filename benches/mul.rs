use clifford::cga_3d::{point, Bivector, Multivector, Vector, Zero, N, N_BAR};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

criterion_main! { mul }
criterion_group! { mul, wedge_n_and_n_bar, translation }

// fn mul_ty(crit: &mut Criterion) {
//     use clifford::pga_3d::point;
//
//     let a = point(0., 0., 0.);
//     let b = point(1., 1., 1.);
//     let c = point(2., 3., 5.);
//
//     let ab = a.wedge(b);
//     let ac = a.wedge(c);
//
//     crit.bench_function("mul_ty", |b| {
//         b.iter(|| {
//             black_box(ab * ac);
//         })
//     });
// }

pub type Translator = Multivector<f64, Zero, Bivector, Zero, Zero, Zero>;

pub fn translator(x: f64, y: f64, z: f64) -> (Translator, Translator) {
    let t = Vector::new(x, y, z, 0., 0.);
    let b = 0.5 * t * N_BAR;

    // this works because N_BAR * N_BAR = 0 for any t
    // so the reversed bivector sign is enough to make them cancel out
    (1. - b, 1. + b)
}

fn wedge_n_and_n_bar(crit: &mut Criterion) {
    let p = point(2., 3., 5.);
    let (t, t_) = translator(1., 2., 3.);

    crit.bench_function("mul n and n_bar", |b| {
        b.iter(|| {
            black_box(N * N_BAR);
        })
    });
}

fn translation(crit: &mut Criterion) {
    let p = point(2., 3., 5.);
    let (t, t_) = translator(1., 2., 3.);

    crit.bench_function("translate", |b| {
        b.iter(|| {
            black_box((t * p * t_).1);
        })
    });
}
