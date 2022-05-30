extern crate proc_macro;

mod algebra;
mod code_gen;
mod code_gen_mv;

use algebra::Algebra;
use syn::parse::{Parse, ParseStream};
use syn::parse_macro_input;

#[proc_macro]
pub fn clifford(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let algebra = parse_macro_input!(input as Algebra);
    algebra.define().into()
}

#[proc_macro]
pub fn clifford_mv(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let algebra = parse_macro_input!(input as Algebra);
    algebra.define_mv().into()
}

impl Parse for Algebra {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        use syn::token::Comma;
        use syn::LitInt;

        let one: LitInt = input.parse()?;
        let _comma: Comma = input.parse()?;
        let neg_one: LitInt = input.parse()?;
        let _comma: Comma = input.parse()?;
        let zero: LitInt = input.parse()?;

        let one = one.base10_parse()?;
        let neg_one = neg_one.base10_parse()?;
        let zero = zero.base10_parse()?;
        Ok(Algebra::new(one, neg_one, zero))
    }
}
