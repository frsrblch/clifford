extern crate proc_macro;

use crate::algebra::Algebra;

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
