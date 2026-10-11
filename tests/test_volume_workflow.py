import unittest
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gammaincc
from scipy.stats import norm

from sample_data import example_tables, validate_table, effective_volume, model_arrays, guideline_lrv_pairs
from count_models import population_quantiles, validate_payload


class VolumeDataTests(unittest.TestCase):
    def setUp(self):
        self.ins, self.outs = example_tables()

    def test_guideline_example_and_fractional_concentration(self):
        out = validate_table(self.outs, 'out')
        self.assertEqual(out.effective_volume_ml.iloc[0], 1875)
        self.assertEqual(out.concentration_100ml.iloc[0], .16)
        self.ins.loc[0, 'value'] = .25
        self.assertEqual(validate_table(self.ins,'in').concentration_100ml.iloc[0], .25)

    def test_dilution_and_effective_mode(self):
        row = self.outs.iloc[0].copy()
        row.dilution_factor = 10
        self.assertEqual(effective_volume(row),187.5)
        row.volume_method = 'effective'
        row.effective_ml = 1875
        self.assertEqual(effective_volume(row),1875)
        row.volume_method = 'direct'
        self.assertEqual(effective_volume(row),5)

    def test_zero_not_missing_or_floor_in_likelihood(self):
        self.outs.loc[0,'value'] = 0
        self.outs.loc[1,'result_type'] = 'missing'
        out = validate_table(self.outs,'out')
        self.assertEqual(out.iloc[0].value,0)
        self.assertEqual(out.iloc[0].zero_convention_100ml,1/18.75)
        self.assertEqual(out.iloc[1].sample_id,'OUT-3')
        arrays = model_arrays(validate_table(self.ins,'in'),out)
        self.assertEqual(arrays['vals_ab'][0],0)
        self.assertEqual(len(arrays['vals_ab']),11)

    def test_guideline_lrv_uses_pair_ids_and_pools_replicates(self):
        self.ins['pair_id'] = ''
        self.outs['pair_id'] = ''
        self.ins.loc[:1, 'pair_id'] = ['P-1', 'P-2']
        self.outs.loc[:1, 'pair_id'] = ['P-1', 'P-1']
        self.outs.loc[1, 'sample_id'] = 'OUT-1'
        self.outs.loc[1, 'replicate_id'] = '2'
        inflow = validate_table(self.ins, 'in')
        outflow = validate_table(self.outs, 'out')
        paired = guideline_lrv_pairs(inflow, outflow)
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired.iloc[0]['count'], 12)
        self.assertEqual(paired.iloc[0]['effective_volume_ml'], 3750)
        expected = np.log10(8_000_000 / (12 / 37.5))
        self.assertAlmostEqual(paired.iloc[0]['guideline_lrv'], expected)

    def test_reference_factor_raises_influent_instead_of_scaling_the_effluent(self):
        """Kap. 4.6.1: the influent is lifted onto V_r; the zero count stays put."""
        self.ins.loc[0, 'pair_id'] = 'P-1'
        self.outs.loc[0, 'pair_id'] = 'P-1'
        self.outs.loc[0, 'value'] = 0
        paired = guideline_lrv_pairs(validate_table(self.ins, 'in'), validate_table(self.outs, 'out'))
        row = paired.iloc[0]
        self.assertEqual(row['reference_factor'], 18.75)
        self.assertEqual(row['inflow_concentration_reference'], 8_000_000 * 18.75)
        # Eq. 4-3 over the floored count, and the per-100-mL form, are the same ratio.
        self.assertAlmostEqual(row['guideline_lrv'], np.log10(row['inflow_concentration_reference'] / 1.))
        self.assertAlmostEqual(row['guideline_lrv'],
                               np.log10(row['inflow_concentration_100ml'] / row['effluent_denominator_100ml']))
        # The enrichment, not the zero, is what buys the extra orders of magnitude.
        self.assertAlmostEqual(row['guideline_lrv'] - np.log10(8_000_000), np.log10(18.75))

    def test_guideline_lrv_zero_uses_one_count_over_total_exposure(self):
        self.ins.loc[0, 'pair_id'] = 'P-1'
        self.outs.loc[0, 'pair_id'] = 'P-1'
        self.outs.loc[0, 'value'] = 0
        paired = guideline_lrv_pairs(validate_table(self.ins, 'in'), validate_table(self.outs, 'out'))
        self.assertAlmostEqual(paired.iloc[0]['effluent_denominator_100ml'], 1 / 18.75)
        self.assertAlmostEqual(paired.iloc[0]['guideline_lrv'], np.log10(8_000_000 * 18.75))

    def test_paired_model_arrays_align_influent_with_effluent_samples(self):
        """Conditional candidates need one influent value per effluent sample, in group order."""
        self.outs = self.outs.iloc[::-1].reset_index(drop=True)
        self.ins['value'] = [10.**(3 + k) for k in range(len(self.ins))]
        inflow, outflow = validate_table(self.ins, 'in'), validate_table(self.outs, 'out')
        arrays = model_arrays(inflow, outflow, paired=True)
        groups = np.asarray(arrays['sample_groups'])
        self.assertEqual(len(arrays['vals_zu']), len(np.unique(groups)))
        lookup = inflow.set_index('pair_id').concentration_100ml
        for group, (_, row) in enumerate(outflow.drop_duplicates('sample_id').iterrows()):
            self.assertEqual(arrays['vals_zu'][group], float(lookup[row.pair_id]))
        validate_payload(dict(family='conditional_poisson_lognormal', **arrays))

    def test_paired_model_arrays_drop_unmatched_influent_and_pool_replicates(self):
        historical = self.ins.iloc[[0]].copy()
        historical['sample_id'], historical['pair_id'] = 'IN-HIST', ''
        self.ins = pd.concat([self.ins, historical], ignore_index=True)
        replicate = self.outs.iloc[[0]].copy()
        replicate['replicate_id'] = '2'
        self.outs = pd.concat([self.outs, replicate], ignore_index=True)
        inflow, outflow = validate_table(self.ins, 'in'), validate_table(self.outs, 'out')
        unpaired = model_arrays(inflow, outflow)
        paired = model_arrays(inflow, outflow, paired=True)
        # The historical influent value stays available to the unpaired candidates only.
        self.assertEqual(len(unpaired['vals_zu']), len(paired['vals_zu']) + 1)
        self.assertEqual(len(paired['vals_ab']), len(outflow))
        self.assertEqual(len(paired['vals_zu']), outflow.sample_id.nunique())
        with self.assertRaises(ValueError):
            validate_payload(dict(family='conditional_poisson_lognormal', **unpaired))

    def test_paired_model_arrays_require_pair_ids(self):
        self.ins['pair_id'] = ''
        inflow, outflow = validate_table(self.ins, 'in'), validate_table(self.outs, 'out')
        model_arrays(inflow, outflow)
        with self.assertRaises(ValueError):
            model_arrays(inflow, outflow, paired=True)

    def test_units(self):
        baseline = validate_table(self.ins,'in').concentration_100ml
        self.ins['value'] *= 10
        self.ins['unit'] = '/L'
        np.testing.assert_allclose(validate_table(self.ins,'in').concentration_100ml,baseline)

    def test_bad_counts_and_volumes(self):
        for field,value in [('value',.5),('value',-1),('value',np.inf),('recovery',50),('recovery',0),
                            ('dilution_factor',0),('filtered_ml',0),('assayed_ml',401),('eluate_ml',np.nan),
                            ('recovery_source',''),('result_type','<LOQ'),('result_type','MPN')]:
            with self.subTest(field=field,value=value):
                table = self.outs.copy()
                table.loc[0,field]=value
                with self.assertRaises(ValueError): validate_table(table,'out')

    def test_replicates_share_group_and_identity(self):
        self.outs.loc[1,'sample_id']='OUT-1'
        self.outs.loc[1,'replicate_id']='2'
        self.outs.loc[1,'pair_id']='P-1'
        arrays=model_arrays(validate_table(self.ins,'in'),validate_table(self.outs,'out'))
        self.assertEqual(arrays['sample_groups'][:2],[0,0])
        self.outs.loc[1,'replicate_id']='1'
        with self.assertRaises(ValueError): validate_table(self.outs,'out')

    def test_blank_numeric_result_is_error_not_zero(self):
        self.outs.loc[0,'value']=np.nan
        with self.assertRaises(ValueError): validate_table(self.outs,'out')

    def test_replicate_volume_sum(self):
        self.outs.loc[1,'sample_id']='OUT-1'
        self.outs.loc[1,'replicate_id']='2'
        self.outs.loc[:1,'assayed_ml']=250.
        with self.assertRaises(ValueError): validate_table(self.outs,'out')


class QuantileTests(unittest.TestCase):
    def test_pln_analytic(self):
        expected=(12-2+norm.ppf(.1)*np.sqrt(.4**2+.7**2))/np.log(10)
        self.assertAlmostEqual(float(population_quantiles(12,.4,2,.7,'poisson_lognormal')),expected)

    def test_gamma_quantile_against_adaptive_integral(self):
        for shape in [.3,2,10]:
            value=float(population_quantiles(12,.4,2,shape,'negative_binomial', nodes=128))
            def cdf(r):
                return quad(lambda z: norm.pdf(z)*gammaincc(shape,shape*np.exp(12+.4*z-r*np.log(10)-2)), -10,10)[0]
            expected=brentq(lambda r:cdf(r)-.1,-10,30)
            self.assertAlmostEqual(value,expected,places=5)

    def test_payload_rejects_misalignment_and_fractional_groups(self):
        payload=dict(vals_zu=[1,2],vals_ab=[0,3],exposures=[1,2])
        validate_payload(payload)
        for change in [dict(exposures=[1]),dict(vals_ab=[0,.5]),dict(sample_groups=[0,.5]),dict(exposures=[0,1])]:
            with self.assertRaises(ValueError): validate_payload({**payload,**change})


if __name__ == '__main__': unittest.main()

class ImportTests(unittest.TestCase):
    def test_csv_roundtrip_preserves_ids_and_volumes(self):
        import io
        from sample_data import editor_table
        ins, outs = example_tables()
        outs.loc[0,'sample_id'] = '001'
        loaded = pd.read_csv(io.StringIO(outs.to_csv(index=False)), dtype={'sample_id':str,'replicate_id':str})
        validated = validate_table(editor_table(loaded),'out')
        self.assertEqual(validated.sample_id.iloc[0],'001')
        self.assertEqual(validated.effective_volume_ml.iloc[0],1875.)
