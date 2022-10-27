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
pub fn g3(_: proc_macro::TokenStream) -> proc_macro::TokenStream {
    Algebra::ga3().define().into()
}
