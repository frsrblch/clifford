use clifford::Wedge;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

criterion_main! { mul }
criterion_group! { mul, wedge_n_and_n_bar }

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

fn wedge_n_and_n_bar(crit: &mut Criterion) {
    use clifford::cga_3d::{N, N_BAR};

    crit.bench_function("mul n and n_bar", |b| {
        b.iter(|| {
            black_box(N * N_BAR);
        })
    });
}
