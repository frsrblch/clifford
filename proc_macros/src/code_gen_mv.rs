use super::algebra::*;
use itertools::iproduct;
use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};
use syn::token::{Brace, For, Impl};
use syn::{parse_quote, GenericParam, ItemImpl, WherePredicate};

impl Algebra {
    pub fn define_mv(self) -> TokenStream {
        let _traits = if self.is_homogenous() {
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

        let grade_products = GradeProducts(self);

        let types = TypeMv::iter(self).map(TypeMv::define);

        //         let impl_grade_products = self
        //             .grades()
        //             .flat_map(|a| TypeMv::iter(self).map(move |lhs| (a, lhs)))
        //             .flat_map(|(a, b)| TypeMv::iter(self).map(move |rhs| (a, b, rhs)))
        //             .map(|(output, lhs, rhs)| {
        //                 let op = ProductOp::Grade(output);
        //
        //                 let output1 = op.output_mv(lhs, rhs, self);
        //
        //                 let product_trait = op.trait_ty();
        //                 let product_fn = op.trait_fn();
        //
        //                 let lhs_ty = lhs.ty_with_suffix("L");
        //                 let rhs_ty = rhs.ty_with_suffix("R");
        //
        //                 let lhs_generics = lhs.generics("L");
        //                 let rhs_generics = rhs.generics("R");
        //
        //                 let mut intermediate_generics = Generics::default();
        //                 let mut where_clause = WhereClause::default();
        //
        //                 if lhs.is_mv() || rhs.is_mv() {
        //                     for (n, (l, r)) in iproduct!(lhs.grades(), rhs.grades())
        //                         .filter(|(l, r)| {
        //                             cartesian_product(*l, *r, op).any(|(l, r, p)| output1.contains(p))
        //                         })
        //                         .enumerate()
        //                     {
        //                         let ident = output.generic(&n.to_string());
        //                         let generic = Generic::Generic(ident.clone());
        //                         intermediate_generics.params.push(generic);
        //
        //                         let lhs_ty = if lhs.is_mv() {
        //                             Generic::Generic(l.generic("L"))
        //                         } else {
        //                             Generic::Concrete(l.ty())
        //                         };
        //
        //                         let rhs_ty = if rhs.is_mv() {
        //                             Generic::Generic(r.generic("R"))
        //                         } else {
        //                             Generic::Concrete(r.ty())
        //                         };
        //
        //                         let bound = Bound {
        //                             ty: lhs_ty,
        //                             trait_ty: product_trait.clone().to_token_stream(),
        //                             generics: Generics::from_iter([rhs_ty, Generic::Output(ident)]),
        //                         };
        //                         where_clause.bounds.push(bound);
        //                     }
        //                 }
        //
        //                 let first = intermediate_generics.iter();
        //                 let next = intermediate_generics.iter().skip(1);
        //                 let len = intermediate_generics.params.len();
        //                 let mut intermediate_sum = Generics::default();
        //                 for (i, (first, next)) in first.zip(next).enumerate() {
        //                     let i = len + i;
        //                     let output = output.generic(&i.to_string());
        //                     let bound = Bound {
        //                         ty: intermediate_sum
        //                             .params
        //                             .last()
        //                             .unwrap_or_else(|| first)
        //                             .clone(),
        //                         trait_ty: quote! { std::ops::Add },
        //                         generics: Generics::from_iter([
        //                             next.clone(),
        //                             Generic::Output(output.clone()),
        //                         ]),
        //                     };
        //                     intermediate_sum.params.push(Generic::Generic(output));
        //                     where_clause.bounds.push(bound);
        //                 }
        //
        //                 let output_is_zero = !ProductOp::Mul.output_contains(lhs, rhs, output);
        //
        //                 let output_ty = if output_is_zero {
        //                     Zero::ty().to_token_stream()
        //                 } else {
        //                     match (lhs, rhs) {
        //                         (TypeMv::Zero(_), _) | (_, TypeMv::Zero(_)) => Zero::ty().to_token_stream(),
        //                         (TypeMv::Grade(_), TypeMv::Grade(_)) => {
        //                             output.type_ident().to_token_stream()
        //                         }
        //                         _ => intermediate_generics
        //                             .iter()
        //                             .chain(intermediate_sum.iter())
        //                             .last()
        //                             .unwrap()
        //                             .to_token_stream(),
        //                     }
        //                 };
        //
        //                 let generics = lhs_generics
        //                     .iter()
        //                     .chain(rhs_generics.iter())
        //                     .chain(intermediate_generics.iter())
        //                     .chain(intermediate_sum.iter())
        //                     .cloned()
        //                     .collect::<Generics>();
        //
        //                 let copy_bounds = lhs_generics
        //                     .iter()
        //                     .chain(rhs_generics.iter())
        //                     .map(|g| Bound {
        //                         ty: Generic::Generic(g.clone()),
        //                         trait_ty: quote! { Copy },
        //                         generics: Default::default(),
        //                     });
        //                 where_clause.bounds.extend(copy_bounds);
        //
        //                 let expr = if output_is_zero {
        //                     OutputExpr::Zero
        //                 } else {
        //                     match (lhs, rhs) {
        //                         (TypeMv::Zero(_), _) | (_, TypeMv::Zero(_)) => OutputExpr::Zero,
        //                         (TypeMv::Grade(_), TypeMv::Grade(_)) => {
        //                             let blades = output
        //                                 .blades()
        //                                 .map(|blade| {
        //                                     let products = lhs
        //                                         .blades()
        //                                         .flat_map(|lhs| rhs.blades().map(move |rhs| (lhs, rhs)))
        //                                         .filter_map(|(lhs_b, rhs_b)| {
        //                                             let product = lhs_b * rhs_b;
        //                                             let out = product.blade()?;
        //
        //                                             if out != blade {
        //                                                 return None;
        //                                             }
        //
        //                                             let l = access_field(lhs, lhs_b, quote!(self));
        //                                             let r = access_field(rhs, rhs_b, quote!(rhs));
        //
        //                                             if product.is_pos() {
        //                                                 Some(parse_quote! { #l * #r })
        //                                             } else {
        //                                                 Some(parse_quote! { -(#l * #r) })
        //                                             }
        //                                         })
        //                                         .collect();
        //
        //                                     (blade, products)
        //                                 })
        //                                 .collect();
        //                             OutputExpr::GradeFields(output, blades)
        //                         }
        //                         _ => {
        //                             let sum = lhs
        //                                 .grades()
        //                                 .flat_map(|l| rhs.grades().map(move |r| (l, r)))
        //                                 .filter(|(l, r)| ProductOp::Mul.output_contains(*l, *r, output))
        //                                 .map(|(l, r)| {
        //                                     let lg = access_grade(lhs, l, quote!(self));
        //                                     let rg = access_grade(rhs, r, quote!(rhs));
        //                                     parse_quote! { #lg.#product_fn(#rg) }
        //                                 })
        //                                 .collect();
        //                             OutputExpr::GradeSum(sum)
        //                         }
        //                     }
        //                 };
        //
        //                 let rhs_ident = if output_is_zero {
        //                     quote! {_}
        //                 } else {
        //                     quote! {rhs}
        //                 };
        //
        //                 quote! {
        //                     impl #generics #product_trait<#rhs_ty> for #lhs_ty #where_clause {
        //                         type Output = #output_ty;
        //                         fn #product_fn(self, #rhs_ident: #rhs_ty) -> Self::Output {
        //                             #expr
        //                         }
        //                     }
        //                 }
        //             });

        quote! {
            // #traits
            #grade_products
            #(#types)*
            // #(#impl_grade_products)*
        }
    }
}

#[allow(dead_code)]
pub fn impl_item_for_product_op(op: ProductOp, lhs: TypeMv, rhs: TypeMv) -> ItemImpl {
    let lhs_suffix = "Lhs";
    let rhs_suffix = "Rhs";
    let out_suffix = "Out";

    let algebra = lhs.algebra();
    let output = op.output_mv(lhs, rhs, algebra);

    let mut generics = syn::Generics::default();
    let mut grade_counter = vec![0usize; algebra.grades().count()];

    fn next_ident(grade_counter: &mut [usize], grade: Grade) -> Ident {
        let counter = &mut grade_counter[grade.0 as usize];
        let n = *counter;
        *counter += 1;
        grade.generic_n(n)
    }

    let trait_ = op.ty();
    let lhs_ty = lhs.ty_with_suffix(lhs_suffix);
    let rhs_ty = rhs.ty_with_suffix(rhs_suffix);
    let trait_ = Some((None, parse_quote!(#trait_<#rhs_ty>), For::default()));

    let op_fn = op.fn_ident();

    let output_expr = match output {
        TypeMv::Zero(_) => Zero::expr(),
        TypeMv::Grade(grade) => {
            let fields = grade
                .blades()
                .map(|blade| {
                    let sum = iproduct!(lhs.blades(), rhs.blades())
                        .filter_map(|(lhs_blade, rhs_blade)| {
                            op.expr(lhs, lhs_blade, rhs, rhs_blade, blade)
                        })
                        .collect();
                    (blade, sum)
                })
                .collect();
            OutputExpr::GradeFields(grade, fields).expr()
        }
        TypeMv::Multivector(mv) => {
            let grades = algebra
                .grades()
                .map(|grade| {
                    let op_fn = ProductOp::Grade(grade).fn_ident();
                    let op_trait = ProductOp::Grade(grade).ty();
                    let sum = iproduct!(lhs.grades(), rhs.grades())
                        .filter_map(|(lhs_grade, rhs_grade)| {
                            if op.output_contains(lhs_grade, rhs_grade, grade) {
                                if output.is_generic() {
                                    generics.make_where_clause().predicates.push({
                                        let lhs = lhs_grade.generic(lhs_suffix);
                                        let rhs = rhs_grade.generic(rhs_suffix);
                                        let out = next_ident(&mut grade_counter, grade);
                                        parse_quote!( #lhs: #op_trait<#rhs, Output = #out> )
                                    });
                                }

                                let lhs = access_grade(lhs, lhs_grade, quote!(self));
                                let rhs = access_grade(rhs, rhs_grade, quote!(rhs));
                                Some(parse_quote!( #lhs.#op_fn(#rhs) ))
                            } else {
                                None
                            }
                        })
                        .collect();
                    (grade, sum)
                })
                .collect();
            OutputExpr::Multivector(grades).expr()

            // let ident = Multivector::ident();
            // let fields = algebra.grades().map(|grade| {
            //     let eq_grade = |g: Grade| g == grade;
            //     if mv.grades().any(eq_grade) {
            //         let op_fn = ProductOp::Grade(grade).fn_ident();
            //         let op_trait = ProductOp::Grade(grade).ty();
            //         let sum = iproduct!(lhs.grades(), rhs.grades()).filter_map::<syn::Expr, _>(
            //             |(lhs_grade, rhs_grade)| {
            //                 if op.output_contains(lhs_grade, rhs_grade, grade) {
            //                     if output.is_generic() {
            //                         generics.make_where_clause().predicates.push({
            //                             let lhs = lhs_grade.generic(lhs_suffix);
            //                             let rhs = rhs_grade.generic(rhs_suffix);
            //                             let out = next_ident(&mut grade_counter, grade);
            //                             parse_quote!( #lhs: #op_trait<#rhs, Output = #out> )
            //                         });
            //                     }
            //
            //                     let lhs = access_grade(lhs, lhs_grade, quote!(self));
            //                     let rhs = access_grade(rhs, rhs_grade, quote!(rhs));
            //                     Some(parse_quote!( #lhs.#op_fn(#rhs) ))
            //                 } else {
            //                     None
            //                 }
            //             },
            //         );
            //
            //         parse_quote!( #(#sum)+* )
            //     } else {
            //         Zero::expr()
            //     }
            // });
            //
            // parse_quote! {
            //     #ident ( #(#fields),* )
            // }
        }
    };

    let trait_fn = parse_quote! {
        #[inline]
        #[allow(unused_variables)]
        fn #op_fn(self, rhs: #rhs_ty) -> Self::Output {
            #output_expr
        }
    };

    generics.params.extend(
        lhs.generics(lhs_suffix)
            .into_iter()
            .chain(rhs.generics(rhs_suffix))
            .map::<GenericParam, _>(|ident| parse_quote!(#ident)),
    );
    generics.make_where_clause().predicates.extend(
        lhs.generics(lhs_suffix)
            .into_iter()
            .chain(rhs.generics(rhs_suffix))
            .map::<WherePredicate, _>(|ident| parse_quote!(#ident: Copy)),
    );

    if output.is_generic() {
        for (g, n) in grade_counter.iter().enumerate() {
            let grade = algebra.grade(g as u8);
            for n in 0..*n {
                let ident = grade.generic_n(n);
                generics.params.push(parse_quote!(#ident));
            }
        }

        for (g, n) in grade_counter.clone().into_iter().enumerate() {
            let grade = algebra.grade(g as u8);
            let max = n.checked_sub(1).unwrap_or_default();
            let mut prev = None;
            for n in 0..max {
                let ident = prev.unwrap_or_else(|| grade.generic_n(0));
                let next = grade.generic_n(n + 1);
                let out = next_ident(&mut grade_counter, grade);
                prev = Some(out.clone());

                let predicates = &mut generics.make_where_clause().predicates;
                predicates.push(parse_quote!( #ident: std::ops::Add<#next, Output = #out>));
                generics.params.push(parse_quote!(#out));
            }
        }
    }

    let output_ty: syn::Type = {
        match output {
            TypeMv::Multivector(mv) if mv.is_generic() => {
                let ident = Multivector::ident();
                let types = grade_counter
                    .iter()
                    .enumerate()
                    .map::<syn::Type, _>(|(g, n)| {
                        let grade = algebra.grade(g as u8);
                        n.checked_sub(1)
                            .map(|n| {
                                let ident = grade.generic_n(n);
                                parse_quote!(#ident)
                            })
                            .unwrap_or_else(Zero::ty)
                    });
                parse_quote!( #ident <#(#types),*> )
            }
            _ => output.ty_with_suffix(out_suffix),
        }
    };
    let type_output = parse_quote! { type Output = #output_ty; };

    ItemImpl {
        attrs: vec![],
        defaultness: None,
        unsafety: None,
        impl_token: Impl::default(),
        generics,
        trait_,
        self_ty: Box::new(lhs_ty),
        brace_token: Brace::default(),
        items: vec![type_output, trait_fn],
    }
}

pub enum OutputExpr {
    Zero,
    GradeFields(Grade, Vec<(Blade, Vec<syn::Expr>)>),
    GradeSum(Vec<syn::Expr>),
    Multivector(Vec<(Grade, Vec<syn::Expr>)>),
}

impl OutputExpr {
    pub fn expr(&self) -> syn::Expr {
        match self {
            Self::Zero => {
                let ty = Zero::ty();
                parse_quote! { #ty }
            }
            Self::GradeFields(grade, blades) => {
                if grade.is_scalar() {
                    let expr = assign_field(TypeMv::Grade(*grade), blades[0].0, &blades[0].1);
                    parse_quote!(#expr)
                } else {
                    let ty = grade.ident();
                    let fields = blades
                        .iter()
                        .map(|(blade, ts)| assign_field(TypeMv::Grade(*grade), *blade, ts));
                    parse_quote! {
                        #ty {
                            #( #fields, )*
                        }
                    }
                }
            }
            Self::GradeSum(sum) => {
                parse_quote! {
                    #(#sum)+*
                }
            }
            Self::Multivector(grades) => {
                let ident = Multivector::ident();
                let algebra = grades.iter().next().unwrap().0 .1;
                let fields = algebra.grades().map(|grade| {
                    if let Some((_, sum)) = grades.iter().find(|(g, _)| *g == grade) {
                        if sum.is_empty() {
                            Zero::expr()
                        } else {
                            parse_quote! {
                                #( #sum )+*
                            }
                        }
                    } else {
                        Zero::expr()
                    }
                });
                parse_quote! {
                    #ident(
                        #(#fields),*
                    )
                }
            }
        }
    }
}

impl ToTokens for OutputExpr {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.expr().to_tokens(tokens);
    }
}

fn access_grade(parent: TypeMv, grade: Grade, ident: TokenStream) -> syn::Expr {
    let field = match parent {
        TypeMv::Grade(_) => return parse_quote! { #ident },
        TypeMv::Multivector(_) => grade.mv_field(),
        TypeMv::Zero(_) => unreachable!("no grades to access"),
    };
    parse_quote! {
        #ident.#field
    }
}

fn assign_field(output: TypeMv, blade: Blade, sum: &Vec<syn::Expr>) -> TokenStream {
    let expr = if sum.is_empty() {
        quote! { 0. }
    } else {
        quote! { #( #sum )+* }
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

            let ty = grade.ident();

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
                let ty = grade.ident();
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
                let generics = mv.type_parameters("");

                let fields = generics.iter().map(|g| {
                    quote! {
                        pub #g,
                    }
                });

                quote! {
                    #[derive(Debug, Default, Copy, Clone, PartialEq)]
                    pub struct Multivector #(#generics),* ( #(#fields)* );
                }
            }
        }
    }

    pub fn ty_with_suffix(&self, suffix: &str) -> syn::Type {
        match self {
            Self::Zero(_) => Zero::ty(),
            Self::Grade(g) => g.ty(),
            Self::Multivector(mv) => {
                let ty = Multivector::ident();
                let generics = mv.type_parameters(suffix);
                parse_quote! {
                    #ty < #(#generics),* >
                }
            }
        }
    }
}

impl Multivector {
    const NAME: &'static str = "Multivector";

    pub fn ident() -> syn::Ident {
        syn::parse_str(Self::NAME).unwrap()
    }
}

impl Grade {
    pub fn generic(&self, suffix: &str) -> Ident {
        let g = self.0;
        Ident::new(&format!("G{g}{suffix}"), Span::mixed_site())
    }

    pub fn generic_n(&self, n: usize) -> Ident {
        let g = self.0;
        Ident::new(&format!("G{g}_{n}"), Span::mixed_site())
    }

    pub fn mv_field(&self) -> syn::Member {
        syn::Member::Unnamed(syn::Index::from(self.0 as usize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_to_file(tokens: &impl ToTokens) {
        let contents = tokens.to_token_stream().to_string();
        let file = "../output.rs";
        std::fs::write(file, contents).unwrap();
    }

    #[test]
    fn print_to_file() {
        let algebra = Algebra::new(4, 1, 0);
        let mv = TypeMv::Multivector(Multivector::new(algebra));

        let impl_item = impl_item_for_product_op(ProductOp::Mul, mv, mv);

        write_to_file(&impl_item);
    }

    #[test]
    fn mv_mv_product() {
        let algebra = Algebra::new(2, 0, 0);
        let mv = TypeMv::Multivector(Multivector::new(algebra));

        let impl_item = impl_item_for_product_op(ProductOp::Mul, mv, mv);

        assert_eq!(
            impl_item.to_token_stream().to_string(),
            quote! {
            impl < G0Lhs , G1Lhs , G2Lhs , G0Rhs , G1Rhs , G2Rhs , G0_0 , G0_1 , G0_2 , G1_0 , G1_1 , G1_2 , G1_3 , G2_0 , G2_1 , G2_2 , G0_3 , G0_4 , G1_4 , G1_5 , G1_6 , G2_3 , G2_4 > std :: ops :: Mul < Multivector < G0Rhs , G1Rhs , G2Rhs > > for Multivector < G0Lhs , G1Lhs , G2Lhs > where G0Lhs : ScalarProduct < G0Rhs , Output = G0_0 > , G1Lhs : ScalarProduct < G1Rhs , Output = G0_1 > , G2Lhs : ScalarProduct < G2Rhs , Output = G0_2 > , G0Lhs : VectorProduct < G1Rhs , Output = G1_0 > , G1Lhs : VectorProduct < G0Rhs , Output = G1_1 > , G1Lhs : VectorProduct < G2Rhs , Output = G1_2 > , G2Lhs : VectorProduct < G1Rhs , Output = G1_3 > , G0Lhs : BivectorProduct < G2Rhs , Output = G2_0 > , G1Lhs : BivectorProduct < G1Rhs , Output = G2_1 > , G2Lhs : BivectorProduct < G0Rhs , Output = G2_2 > , G0Lhs : Copy , G1Lhs : Copy , G2Lhs : Copy , G0Rhs : Copy , G1Rhs : Copy , G2Rhs : Copy , G0_0 : std :: ops :: Add < G0_1 , Output = G0_3 > , G0_3 : std :: ops :: Add < G0_2 , Output = G0_4 > , G1_0 : std :: ops :: Add < G1_1 , Output = G1_4 > , G1_4 : std :: ops :: Add < G1_2 , Output = G1_5 > , G1_5 : std :: ops :: Add < G1_3 , Output = G1_6 > , G2_0 : std :: ops :: Add < G2_1 , Output = G2_3 > , G2_3 : std :: ops :: Add < G2_2 , Output = G2_4 > { type Output = Multivector < G0_4 , G1_6 , G2_4 > ; # [inline] # [allow (unused_variables)] fn mul (self , rhs : Multivector < G0Rhs , G1Rhs , G2Rhs >) -> Self :: Output { Multivector (self . 0 . scalar_prod (rhs . 0) + self . 1 . scalar_prod (rhs . 1) + self . 2 . scalar_prod (rhs . 2) , self . 0 . vector_prod (rhs . 1) + self . 1 . vector_prod (rhs . 0) + self . 1 . vector_prod (rhs . 2) + self . 2 . vector_prod (rhs . 1) , self . 0 . bivector_prod (rhs . 2) + self . 1 . bivector_prod (rhs . 1) + self . 2 . bivector_prod (rhs . 0)) } }
        }
                .to_string()
        );
    }

    #[test]
    fn vector_product_of_two_vectors_is_zero() {
        let algebra = Algebra::new(2, 0, 0);
        let vector = TypeMv::Grade(algebra.grade(1));

        // <vector * vector>_1 = 0
        let impl_item =
            impl_item_for_product_op(ProductOp::Grade(algebra.grade(1)), vector, vector);

        assert_eq!(
            impl_item.to_token_stream().to_string(),
            quote! {
            impl VectorProduct < Vector > for Vector { type Output = Zero ; # [inline] # [allow (unused_variables)] fn vector_prod (self , rhs : Vector) -> Self :: Output { Zero } }
        }.to_string()
        );
    }

    #[test]
    fn vector_product_of_vectors_and_bivector_is_vector() {
        let algebra = Algebra::new(3, 0, 0);
        let vector = TypeMv::Grade(algebra.grade(1));
        let bivector = TypeMv::Grade(algebra.grade(2));

        // <vector * bivector>_1 = vector
        let impl_item =
            impl_item_for_product_op(ProductOp::Grade(algebra.grade(1)), vector, bivector);

        assert_eq!(
            impl_item.to_token_stream().to_string(),
            quote! {
            impl VectorProduct < Bivector > for Vector { type Output = Vector ; # [inline] # [allow (unused_variables)] fn vector_prod (self , rhs : Bivector) -> Self :: Output { Vector { e1 : - (self . e2 * rhs . e12) + - (self . e3 * rhs . e13) , e2 : self . e1 * rhs . e12 + - (self . e3 * rhs . e23) , e3 : self . e1 * rhs . e13 + self . e2 * rhs . e23 , } } }
        }.to_string()
        );
    }

    #[test]
    fn vector_vector_product() {
        let algebra = Algebra::new(3, 0, 0);
        let vector = TypeMv::Grade(algebra.grade(1));

        let impl_item = impl_item_for_product_op(ProductOp::Mul, vector, vector);

        // write_to_file(&impl_item);

        assert_eq!(
            impl_item.to_token_stream().to_string(),
            quote! {
                impl std::ops::Mul<Vector> for Vector {
                    type Output = Multivector<f64, Zero, Bivector, Zero>;
                    #[inline]
                    #[allow(unused_variables)]
                    fn mul(self, rhs: Vector) -> Self::Output {
                        Multivector(self.scalar_prod(rhs), Zero, self.bivector_prod(rhs), Zero)
                    }
                }
            }
            .to_string()
        );
    }
}
