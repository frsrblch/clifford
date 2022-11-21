extern crate proc_macro;

use crate::algebra::{Algebra, Basis};

mod algebra;
mod code_gen;
mod dynamic_types;

#[proc_macro]
pub fn pga3(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    Algebra::pga3().define().into()
}

#[proc_macro]
pub fn dyn_pga3(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    Algebra::pga3().dynamic_types().into()
}

#[proc_macro]
pub fn ga3(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    Algebra::ga3().define().into()
}

#[proc_macro]
pub fn dyn_ga3(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    Algebra::ga3().dynamic_types().into()
}

#[proc_macro]
pub fn pos_vel_ga(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let bases = Box::new([
        Basis::pos('x'),
        Basis::pos('y'),
        Basis::pos('z'),
        Basis::pos('t'),
        Basis::pos('u'),
        Basis::pos('v'),
    ]);
    Algebra {
        bases: Box::leak(bases),
        slim: true,
    }
    .define()
    .into()
}
