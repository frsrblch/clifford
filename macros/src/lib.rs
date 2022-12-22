extern crate proc_macro;

use syn::{parse::Parse, parse_macro_input, token::Comma, LitInt};

use crate::algebra::{Algebra, Basis};

mod algebra;
mod code_gen;
mod dynamic_types;

#[proc_macro]
#[allow(non_snake_case)]
pub fn algebra(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let algebra = parse_macro_input!(input as Algebra);
    algebra.define().into()
}

#[proc_macro]
#[allow(non_snake_case)]
pub fn algebra_slim(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let mut algebra = parse_macro_input!(input as Algebra);
    algebra.slim = true;
    algebra.define().into()
}

impl Parse for Algebra {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let pos: u32 = input.parse::<LitInt>()?.base10_parse()?;

        if input.is_empty() {
            return Ok(Algebra::from(IntAlgebra { pos, neg: 0, zero: 0 }));
        }

        let _: Comma = input.parse()?;

        let neg: u32 = input.parse::<LitInt>()?.base10_parse()?;

        if input.is_empty() {
            return Ok(Algebra::from(IntAlgebra { pos, neg, zero: 0 }));
        }

        let _: Comma = input.parse()?;

        let zero = input.parse::<LitInt>()?.base10_parse()?;

        Ok(Algebra::from(IntAlgebra { pos, neg, zero }))
    }
}

pub(crate) struct IntAlgebra {
    pos: u32,
    neg: u32,
    zero: u32,
}

impl From<IntAlgebra> for Algebra {
    fn from(a: IntAlgebra) -> Self {
        let mut bases = Vec::with_capacity((a.pos + a.neg + a.zero) as usize);
        for p in 0..a.pos {
            bases.push(Basis {
                char: std::char::from_digit(p + 1, 16).expect("std::char::from_digit out of range"),
                sqr: algebra::Square::Pos,
            });
        }
        for q in 0..a.neg {
            bases.push(Basis {
                char: std::char::from_digit(q + a.pos + 1, 16)
                    .expect("std::char::from_digit out of range"),
                sqr: algebra::Square::Neg,
            });
        }
        for z in 0..a.zero {
            bases.push(Basis {
                char: std::char::from_digit(z + a.pos + a.neg + 1, 16)
                    .expect("std::char::from_digit out of range"),
                sqr: algebra::Square::Zero,
            });
        }
        Algebra {
            bases: Box::leak(bases.into()),
            slim: false,
        }
    }
}

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

#[cfg(test)]
mod tests {
    use crate::algebra::Square;

    use super::*;

    #[test]
    fn parse_test() {
        let g3: Algebra = syn::parse_str("3").unwrap();
        assert_eq!(3, g3.bases.len());
        assert!(g3.bases.iter().all(|b| b.sqr == Square::Pos));

        let cga3: Algebra = syn::parse_str("4, 1").unwrap();
        assert_eq!(5, cga3.bases.len());
        assert_eq!(
            4,
            cga3.bases.iter().filter(|b| b.sqr == Square::Pos).count()
        );
        assert_eq!(
            1,
            cga3.bases.iter().filter(|b| b.sqr == Square::Neg).count()
        );

        let pga3: Algebra = syn::parse_str("3, 0, 1").unwrap();
        assert_eq!(4, pga3.bases.len());
        assert_eq!(
            3,
            pga3.bases.iter().filter(|b| b.sqr == Square::Pos).count()
        );
        assert_eq!(
            1,
            pga3.bases.iter().filter(|b| b.sqr == Square::Zero).count()
        );
    }
}
