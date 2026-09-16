import copy
import unittest
from types import SimpleNamespace
from lab.rubin_two_node.validate_actor_batch import validate_samples


def samples():
    return [SimpleNamespace(index=i,group_index=i//8,rollout_id=None,status=SimpleNamespace(value='completed'),
            remove_sample=False,tokens=[10+i//8,20,30],response_length=1,response='#### 4',label='4',
            prompt='question '+str(i//8),rollout_log_probs=[-0.5],loss_mask=None,reward={'reward':1},
            validate=lambda:None) for i in range(2048)]


class BatchTests(unittest.TestCase):
    def test_real_shape_grouping_and_implicit_masks_are_accepted(self):
        data = samples()
        before = copy.deepcopy([x.loss_mask for x in data])
        result = validate_samples(data,scorer=lambda response,label:{'reward':1})
        self.assertEqual(result['sample_count'],2048)
        self.assertEqual(result['prompt_groups'],256)
        self.assertEqual(result['mask_counts'],{'implicit_all_ones':2048})
        self.assertEqual([x.loss_mask for x in data],before)

    def test_corrupt_logprob_mask_status_reward_and_group_are_refused(self):
        changes = [('rollout_log_probs',[float('nan')],'log probability'),
                   ('loss_mask',[2],'loss mask'),('status',SimpleNamespace(value='aborted'),'terminal status'),
                   ('reward',{'reward':float('inf')},'reward'),('group_index',999,'group')]
        for field,value,reason in changes:
            with self.subTest(field=field):
                data=samples()
                setattr(data[0],field,value)
                with self.assertRaisesRegex(ValueError,reason): validate_samples(data)

    def test_native_group_reordering_allowed_but_interleaved_members_refused(self):
        data=samples()
        reordered=data[8:16]+data[:8]+data[16:]
        self.assertEqual(validate_samples(reordered)['prompt_groups'],256)
        data[1],data[8]=data[8],data[1]
        with self.assertRaisesRegex(ValueError,'not contiguous'): validate_samples(data)

    def test_same_shape_different_token_fingerprint_and_reward_recompute(self):
        data=samples()
        first=validate_samples(data)['ordered_sample_fingerprint']
        data[0].tokens[-1]=77
        self.assertNotEqual(validate_samples(data)['ordered_sample_fingerprint'],first)
        with self.assertRaisesRegex(ValueError,'disagrees'):
            validate_samples(data,scorer=lambda response,label:{'reward':0})

    def test_count_duplicate_index_and_group_prompt_mismatch_refused(self):
        data=samples()
        with self.assertRaisesRegex(ValueError,'Expected 2048'): validate_samples(data[:-1])
        data[1].index=0
        with self.assertRaisesRegex(ValueError,'duplicate'): validate_samples(data)
        data=samples()
        data[1].tokens[0]=999
        with self.assertRaisesRegex(ValueError,'group prompt'): validate_samples(data)


if __name__=='__main__': unittest.main()
