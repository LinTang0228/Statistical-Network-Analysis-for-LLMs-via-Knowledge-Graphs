# ICL_prompt.py

""" These findings support previous studies that showed that the use of aspirin during the antecedent illness may be a risk factor for the development of RS . """
extract = Extract([Triple(adverse_effect('RS'), Adverse_Effect(), drug('aspirin')),])
""" A 34-year - old lady developed a constellation of dermatitis , fever , lymphadenopathy and hepatitis , beginning on the 17th day of a course of oral sulphasalazine for sero - negative rheumatoid arthritis . """
extract = Extract([Triple(adverse_effect('constellation of dermatitis'), Adverse_Effect(), drug('sulphasalazine')), Triple(adverse_effect('fever'), Adverse_Effect(), drug('sulphasalazine')), Triple(adverse_effect('hepatitis'), Adverse_Effect(), drug('sulphasalazine')), Triple(adverse_effect('lymphadenopathy'), Adverse_Effect(), drug('sulphasalazine')),])
""" Gemcitabine - induced pulmonary toxicity is usually a dramatic condition . """
extract = Extract([Triple(adverse_effect('pulmonary toxicity'), Adverse_Effect(), drug('Gemcitabine')),])
""" In this article , we describe another case of subcutaneous changes following repeated glatiramer acetate injection , presented as localized panniculitis in the area around the injection sites , in a 46-year - old female patient who was treated with glatiramer acetate for 18 months . """
extract = Extract([Triple(adverse_effect('localized panniculitis'), Adverse_Effect(), drug('glatiramer acetate')), Triple(adverse_effect('subcutaneous changes'), Adverse_Effect(), drug('glatiramer acetate')),])
""" We present the management of agranulocytosis and neutropenic sepsis secondary to carbimazole with recombinant human granulocyte colony stimulating factor ( G - CSF ) . """
extract = Extract([Triple(adverse_effect('agranulocytosis'), Adverse_Effect(), drug('carbimazole')), Triple(adverse_effect('neutropenic sepsis'), Adverse_Effect(), drug('carbimazole')),])
""" We conclude that ( a ) cyclophosphamide is a human teratogen , ( b ) a distinct phenotype exists , and ( c ) the safety of CP in pregnancy is in serious question . """
extract = Extract([Triple(adverse_effect('human teratogen'), Adverse_Effect(), drug('cyclophosphamide')),])