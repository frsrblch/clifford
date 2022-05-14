use super::types::*;
use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};
use syn::punctuated::Punctuated;
use syn::token::{Add, Comma};

impl Algebra {
    pub fn define_mv(&self) -> TokenStream {
        let traits = if self.is_homogenous() {
            quote! {
                pub use crate::{
                    Dot,
                    Antidot,
                    Wedge,
                    Antiwedge,
                    Commutator,
                    Geometric,
                    Antigeometric,
                    LeftComplement,
                    RightComplement,
                    Reverse,
                    Antireverse,
                    Bulk,
                    Weight,
                    IsIdeal
                };
            }
        } else {
            quote! {
                pub use crate::{
                    Dot,
                    Wedge,
                    Commutator,
                    Geometric,
                    LeftComplement,
                    RightComplement,
                    Reverse
                };
            }
        };

        let grade_products = GradeProducts(*self);

        let types = TypeMv::iter(self).map(TypeMv::define);

        let impl_grade_products = self
            .grades()
            .flat_map(|a| TypeMv::iter(self).map(move |lhs| (a, lhs)))
            .flat_map(|(a, b)| TypeMv::iter(self).map(move |rhs| (a, b, rhs)))
            .map(|(output, lhs, rhs)| {
                let product_trait = output.grade_product_trait();
                let product_fn = output.grade_product_fn();

                let lhs_ty = lhs.type_ident();
                let rhs_ty = rhs.type_ident();

                let lhs_generics = lhs.generics("L");
                let rhs_generics = rhs.generics("R");

                let (mut intermediate_generics, mut where_clause) = match (lhs, rhs) {
                    (TypeMv::Multivector(l), TypeMv::Grade(g)) => l
                        .grades()
                        .filter(|l| ProductOp::Mul.output_contains(*l, g, output))
                        .enumerate()
                        .map(|(n, l)| {
                            let ident = output.generic(&n.to_string());
                            let generic = Generic::Generic(ident.clone());
                            let bound = Bound {
                                ty: Box::new(l.generic("L")),
                                trait_ty: product_trait.clone().to_token_stream(),
                                generics: Generics::from_iter([Generic::Concrete(g.type_ident()), Generic::Output(ident)]),
                            };
                            (generic, bound)
                        })
                        .fold((Generics::default(), WhereClause::default()), |(mut gs, mut wc), (g, b)| {
                            gs.params.push(g);
                            wc.bounds.push(b);
                            (gs, wc)
                        }),
                    (TypeMv::Grade(g), TypeMv::Multivector(r)) => r
                        .grades()
                        .filter(|r| ProductOp::Mul.output_contains(g, *r, output))
                        .enumerate()
                        .map(|(n, r)| {
                            let ident = output.generic(&n.to_string());
                            let generic = Generic::Generic(ident.clone());
                            let bound = Bound {
                                ty: Box::new(g.type_ident()),
                                trait_ty: product_trait.clone().to_token_stream(),
                                generics: Generics::from_iter([Generic::Generic(r.generic("R")), Generic::Output(ident)]), 
                            };
                            (generic, bound)
                        })
                        .fold((Generics::default(), WhereClause::default()), |(mut gs, mut wc), (g, b)| {
                            gs.params.push(g);
                            wc.bounds.push(b);
                            (gs, wc)
                        }),

                    (TypeMv::Multivector(l), TypeMv::Multivector(r)) => l
                        .grades()
                        .flat_map(|l| r.grades().map(move |r| (l, r)))
                        .filter(|(l, r)| ProductOp::Mul.output_contains(*l, *r, output))
                        .enumerate()
                        .map(|(n, (l, r))| {
                            let ident = output.generic(&n.to_string());
                            let generic = Generic::Generic(ident.clone());
                            let bound = Bound {
                                ty: Box::new(l.generic("L")),
                                trait_ty: product_trait.clone().to_token_stream(),
                                generics: Generics::from_iter([Generic::Generic(r.generic("R")), Generic::Output(ident)]),
                            };
                            (generic, bound)
                        })
                        .fold((Generics::default(), WhereClause::default()), |(mut gs, mut wc), (g, b)| {
                            gs.params.push(g);
                            wc.bounds.push(b);
                            (gs, wc)
                        }),

                    _ => (Generics::default(), WhereClause::default()),
                };

                let first = intermediate_generics.iter().collect::<Vec<_>>();
                let next = intermediate_generics.iter().skip(1).collect::<Vec<_>>();
                let len = intermediate_generics.params.len();
                let mut intermediate_sum = Generics::default();
                for (i, (first, next)) in first.into_iter().zip(next).enumerate() {
                    let i = len + i;
                    let output = output.generic(&i.to_string());
                    let bound = Bound {
                        ty: Box::new(intermediate_sum.params.last().cloned().unwrap_or_else(|| first.clone())),
                        trait_ty: quote! { std::ops::Add },
                        generics: Generics::from_iter([next.clone(), Generic::Output(output.clone())])
                    };
                    intermediate_sum.params.push(Generic::Generic(output));
                    where_clause.bounds.push(bound);
                }

                let output_zero = !ProductOp::Mul.output_contains(lhs, rhs, output);

                let output_ty = if !output_zero {
                    match (lhs, rhs) {
                        (TypeMv::Zero(_), _) | (_, TypeMv::Zero(_)) => Zero::ty().to_token_stream(),
                        (TypeMv::Grade(_), TypeMv::Grade(_)) => output.type_ident().to_token_stream(),
                        _ => {
                            if let Some(g) = intermediate_generics.iter().chain(intermediate_sum.iter()).last() {
                                quote! { #g }
                            } else {
                                quote! { todo!() }
                            }
                        },
                    }
                } else {
                    Zero::ty().to_token_stream()
                };

                let generics = lhs_generics
                    .iter()
                    .chain(rhs_generics.iter())
                    .chain(intermediate_generics.iter())
                    .chain(intermediate_sum.iter())
                    .cloned()
                    .collect::<Generics>();

                for g in lhs_generics.iter().chain(rhs_generics.iter()) {
                    let bound = Bound {
                        ty: Box::new(g.to_token_stream()),
                        trait_ty: quote! { Copy },
                        generics: Default::default()
                    };
                    where_clause.bounds.push(bound);
                }

                let expr = if output_zero {
                    Zero::ty().to_token_stream()
                } else {
                    match (lhs, rhs) {
                        (TypeMv::Grade(lhs_g), TypeMv::Grade(rhs_g)) => {
                            let fields = output.blades().map(|blade| {
                                let f = blade.field();

                                let fields = lhs
                                    .blades()
                                    .flat_map(|lhs| rhs.blades().map(move |rhs| (lhs, rhs)))
                                    .filter_map(|(lhs_b, rhs_b)| {
                                        let product = lhs_b * rhs_b;
                                        let out = product.blade()?;

                                        if out != blade {
                                            return None;
                                        }

                                        let l = access_value(lhs, lhs_b, quote!(self));
                                        let r = access_value(rhs, rhs_b, quote!(rhs));

                                        if product.is_pos() {
                                            Some(quote! { #l * #r })
                                        } else {
                                            Some(quote! { -(#l * #r) })
                                        }
                                    })
                                    .collect();

                                assign_value(TypeMv::Grade(output), blade, fields)
                            });

                            output_expr(TypeMv::Grade(output), fields)
                        }
                        (TypeMv::Grade(lhs_g), TypeMv::Multivector(rhs_mv)) => {
                            rhs_mv
                                .grades()
                                .filter(|r| ProductOp::Mul.output_contains(lhs_g, *r, output))
                                .map(|r| {
                                    let g = r.mv_field();
                                    quote! { self.#product_fn(rhs.#g)}
                                }).collect::<Punctuated<_, Add>>().to_token_stream()
                        }
                        (TypeMv::Multivector(lhs_mv), TypeMv::Grade(rhs_g)) => {
                            lhs_mv
                                .grades()
                                .filter(|l| ProductOp::Mul.output_contains(*l, rhs_g, output))
                                .map(|l| {
                                    let g = l.mv_field();
                                    quote! { self.#g.#product_fn(rhs)}
                                })
                                .collect::<Punctuated<_, Add>>()
                                .to_token_stream()
                        }
                        (TypeMv::Multivector(lhs_mv), TypeMv::Multivector(rhs_mv)) => {
                            lhs_mv
                                .grades()
                                .flat_map(|l| rhs_mv.grades().map(move |r| (l, r)))
                                .filter(|(l, r)| ProductOp::Mul.output_contains(*l, *r, output))
                                .map(|(l, r)| {
                                    let gl = l.mv_field();
                                    let gr = r.mv_field();
                                    quote! { self.#gl.#product_fn(rhs.#gr) }
                                })
                                .collect::<Punctuated<_, Add>>()
                                .to_token_stream()
                        }
                        (TypeMv::Zero(_), _) | (_, TypeMv::Zero(_)) => Zero::ty().to_token_stream(),
                        _ => quote! { todo!() },
                    }
                };

                let rhs_ident = if output_zero { quote! {_} } else { quote! {rhs} };

                quote! {
                    impl #generics #product_trait<#rhs_ty #rhs_generics> for #lhs_ty #lhs_generics #where_clause {
                        type Output = #output_ty;
                        fn #product_fn(self, #rhs_ident: #rhs_ty #rhs_generics) -> Self::Output {
                            #expr
                        }
                    }
                }
            });

        quote! {
            #traits
            #grade_products
            #(#types)*
            #(#impl_grade_products)*
        }
    }

    fn types_mv(&self) -> impl Iterator<Item = TypeMv> + '_ {
        TypeMv::iter(self)
    }
}

fn output_expr<F: Iterator<Item = TokenStream>>(output: TypeMv, fields: F) -> TokenStream {
    match output {
        TypeMv::Zero(_) => output.type_ident().to_token_stream(),
        TypeMv::Grade(g) => {
            if g.is_scalar() {
                quote! { #( #fields )* }
            } else {
                let ty = output.type_ident();
                quote! {
                    #ty {
                        #( #fields, )*
                    }
                }
            }
        }
        TypeMv::Multivector(mv) => {
            quote! { todo!() }
        }
    }
}

fn access_value(parent: TypeMv, blade: Blade, ident: TokenStream) -> TokenStream {
    if parent.is_scalar() {
        ident
    } else {
        let field = match parent {
            TypeMv::Grade(_) => blade.field().to_token_stream(),
            TypeMv::Multivector(mv) => blade.grade().mv_field(),
            TypeMv::Zero(_) => unreachable!("no fields to access"),
        };

        quote! {
            #ident.#field
        }
    }
}

fn assign_value(output: TypeMv, blade: Blade, sum: Punctuated<TokenStream, Add>) -> TokenStream {
    let expr = if sum.is_empty() {
        quote! { 0. }
    } else {
        sum.to_token_stream()
    };
    if output.is_scalar() {
        expr
    } else {
        let field = blade.field();
        quote! { #field: #expr }
    }
}

pub struct GradeProducts(Algebra);

impl ToTokens for GradeProducts {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        for grade in self.0.grades() {
            let trait_ident = Ident::new(&format!("{}Product", grade.name()), Span::mixed_site());
            let trait_fn = Ident::new(
                &format!("{}_prod", grade.name().to_lowercase()),
                Span::mixed_site(),
            );

            tokens.extend(quote! {
                pub trait #trait_ident<Rhs> {
                    type Output;
                    fn #trait_fn(self, rhs: Rhs) -> Self::Output;
                }
            });
        }
    }
}

pub struct GradeTypes(Algebra);

impl ToTokens for GradeTypes {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        for grade in self.0.grades() {
            if grade.is_scalar() {
                continue;
            }

            let ty = grade.type_ident();

            let fields = grade.blades().map(|blade| {
                let f = blade.field();
                quote! { pub #f: f64, }
            });

            tokens.extend(quote! {
                #[derive(Debug, Default, Copy, Clone, PartialEq)]
                pub struct #ty {
                    #(#fields)*
                }
            })
        }
    }
}

impl TypeMv {
    fn define(self) -> TokenStream {
        match self {
            Self::Zero(_) => quote! {
                #[derive(Debug, Default, Copy, Clone, PartialEq)]
                pub struct Zero;
            },
            Self::Grade(g) if g.is_scalar() => {
                quote! {}
            }
            Self::Grade(grade) => {
                let ty = grade.type_ident();
                let fields = grade.blades().map(|blade| {
                    let f = blade.field();
                    quote! { pub #f: f64, }
                });
                quote! {
                    #[derive(Debug, Default, Copy, Clone, PartialEq)]
                    pub struct #ty {
                        #(#fields)*
                    }
                }
            }
            Self::Multivector(mv) => {
                let generics = mv.generics("");

                let fields = generics.iter().map(|g| {
                    quote! {
                        pub #g,
                    }
                });

                quote! {
                    #[derive(Debug, Default, Copy, Clone, PartialEq)]
                    pub struct Multivector #generics ( #(#fields)* );
                }
            }
        }
    }

    pub fn type_ident(&self) -> syn::Type {
        match self {
            Self::Zero(_) => Zero::ty(),
            Self::Grade(g) => g.type_ident(),
            Self::Multivector(_) => Multivector::type_ident(),
        }
    }
}

impl Multivector {
    const NAME: &'static str = "Multivector";

    pub fn type_ident() -> syn::Type {
        syn::parse_str(Self::NAME).unwrap()
    }
}

impl Grade {
    pub fn generic(&self, suffix: &str) -> Ident {
        let str = match self.0 {
            0 => "V0",
            1 => "V1",
            2 => "V2",
            3 => "V3",
            4 => "V4",
            5 => "V5",
            6 => "V6",
            _ => unimplemented!("not implemented for grade: {}", self.0),
        };
        Ident::new(&format!("{}{}", str, suffix), Span::mixed_site())
    }

    pub fn mv_field(&self) -> TokenStream {
        match self.0 {
            0 => quote! { 0 },
            1 => quote! { 1 },
            2 => quote! { 2 },
            3 => quote! { 3 },
            4 => quote! { 4 },
            5 => quote! { 5 },
            6 => quote! { 6 },
            _ => unimplemented!("not implemented for grade: {}", self.0),
        }
    }

    pub fn grade_product_trait(&self) -> Ident {
        let str = &format!("{}Product", self.name());
        Ident::new(str, Span::mixed_site())
    }

    pub fn grade_product_fn(&self) -> Ident {
        let str = &format!("{}_prod", self.name().to_lowercase());
        Ident::new(str, Span::mixed_site())
    }
}

// fn impl_product_op(lhs: TypeMv, rhs: TypeMv, op: ProductOp, algebra: Algebra) -> TokenStream {
//     let op_trait = op.trait_ty();
//     let op_fn = op.trait_fn();
//
//     let mut intermediate_idents = vec![0u16; algebra.dimensions() as usize + 1];
//     let mut new_ident = |grade: Grade| {
//         let count = &mut intermediate_idents[grade.0 as usize];
//         let n = *count;
//         *count += 1;
//         grade.generic(&n.to_string())
//     };
//     let mut all_generics = Punctuated::<TokenStream, Comma>::default();
//     let lhs_generics = lhs.generics("L");
//     all_generics.extend(lhs_generics.iter().cloned());
//     let rhs_generics = mv_generics(algebra, "R");
//     all_generics.extend(rhs_generics.iter().cloned());
//     let output_generics = mv_generics(algebra, "Out");
//     all_generics.extend(output_generics.iter().cloned());
//
//     let mut trait_bounds = BTreeMap::<String, Punctuated<TokenStream, Add>>::default();
//     trait_bounds.extend(lhs_generics.iter().chain(rhs_generics.iter()).map(|i| {
//         (
//             i.to_string(),
//             std::iter::once_with(|| quote!(Copy)).collect(),
//         )
//     }));
//
//     let mut output_expr = std::collections::HashMap::<Grade, Vec<TokenStream>>::default();
//
//     for lhs in algebra.grades() {
//         let lhs_f = lhs.mv_field();
//         for rhs in algebra.grades() {
//             let rhs_f = rhs.mv_field();
//             for output in algebra.grades() {
//                 if op.output_contains(lhs, rhs, output) {
//                     let output_trait = output.grade_product_trait();
//                     let output_fn = output.grade_product_fn();
//                     let expr = quote! {
//                         self.#lhs_f.#output_fn(rhs.#rhs_f)
//                     };
//                     output_expr.entry(output).or_default().push(expr);
//
//                     let lhs_ident = lhs_generics[lhs.0 as usize].to_token_stream().to_string();
//                     let rhs_ident = rhs_generics[rhs.0 as usize].to_token_stream().to_string();
//                     let output_ident = new_ident(output);
//                     trait_bounds
//                         .entry(lhs_ident)
//                         .or_default()
//                         .push(quote! { #output_trait<#rhs_ident, Output = #output_ident> });
//
//                     all_generics.push(output_ident.to_token_stream());
//                 }
//             }
//         }
//     }
//
//     // TODO replace SumN with std ops and intermediate types
//     for (g, &n) in algebra.grades().zip(intermediate_idents.iter()) {
//         if n > 1 {
//             let sum_trait = Ident::new(&format!("Sum{n}"), Span::mixed_site());
//             let one_to_n = (1..n).into_iter().map(|n| g.generic(&n.to_string()));
//             let output = &output_generics[g.0 as usize];
//             let sum = quote! {
//                 #sum_trait < #(#one_to_n,)* Output = #output>
//             };
//             trait_bounds
//                 .entry(g.generic("0").to_string())
//                 .or_default()
//                 .push(sum);
//         }
//     }
//
//     let trait_bounds = trait_bounds.iter().map(|(i, p)| {
//         quote! { #i: #p, }
//     });
//
//     let fields = algebra.grades().map(|g| {
//         let mut exprs = output_expr.remove(&g).unwrap_or_default();
//         match exprs.len() {
//             0 => Zero::syn_ty().to_token_stream(),
//             1 => exprs.pop().unwrap(),
//             n => {
//                 let first = exprs.remove(0);
//                 let sum_n = Ident::new(&format!("sum{n}"), Span::mixed_site());
//                 let rest = exprs.drain(..).collect::<Punctuated<TokenStream, Comma>>();
//                 quote! {
//                     #first.#sum_n(#rest)
//                 }
//             }
//         }
//     });
//
//     quote! {
//         impl<#all_generics> #op_trait<Multivector<#rhs_generics>> for Multivector<#lhs_generics>
//         where
//             #(#trait_bounds)*
//         {
//             type Output = Multivector<#output_generics>;
//             fn #op_fn(self, rhs: Multivector<#rhs_generics>) -> Self::Output {
//                 Multivector(#(#fields,)*)
//             }
//         }
//     }
// }

fn mv_generics(algebra: Algebra, suffix: &str) -> Punctuated<TokenStream, Comma> {
    algebra
        .grades()
        .map(|g| g.generic(suffix).to_token_stream())
        .collect()
}

#[test]
fn print_to_file() {
    let algebra = Algebra::new(3, 0, 0);
    let contents = algebra.define_mv().to_string();
    let file = "../output.rs";
    std::fs::write(file, contents).unwrap();
}
