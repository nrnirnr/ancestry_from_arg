import msprime, tsinfer, tskit
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from datetime import datetime
import subprocess
from Bio import bgzf
from pathlib import Path
import demesdraw
from scipy.optimize import minimize

eps = 1e-8
from config import RELATE_PATH #TODO: придумать как красиво ее задавать глобально


def to_nodes(samples):
    nodes = np.empty(2 * len(samples))
    nodes[0::2] = 2 * samples
    nodes[1::2] = 2 * samples + 1
    return nodes

## TODO: check if inequal samples counts work (may be difficulties with np.array)
## TODO: check if TS_tsinfer/TS_relate may be defined with no TS_sim template 
##          (create TS_model template just with samples indices, 
##           then initialize TS_tsinfer/TS_relate with the tampolate and existing paths)

class TS_model():
    def __init__(self, anc_samples: np.ndarray | list, targ_samples: np.ndarray | list, 
                 ts: tskit.TreeSequence | None=None, 
                 train_ratio: float=0.4, true_ancestry: np.ndarray | list | None=None):

        anc_samples = np.array(anc_samples, dtype=object)
        targ_samples = np.array(targ_samples)

        N_samples = np.array([len(i) for i in anc_samples] + [len(targ_samples)])

        ids = []
        if N_samples[-1] <= 0:
            raise ValueError('new population needs to contain at least one sample')
        
        for i in range(len(N_samples) - 1):
            if N_samples[i] < 0:
                raise ValueError('populations cannot contain negative samples amount')
            if N_samples[i] > 0:
                ids.append(i)
        if len(ids) == 0:
            raise ValueError('at least one of the ancestral populations needs to contain samples')
        ids.append(len(N_samples) - 1)

        self.ids = ids[:-1] #indices of non empty populations
        self.N_samples = N_samples[ids] #array with numbers of samples for each population, the last element refers to the new population
        self.anc_pops = len(self.N_samples) - 1 #number of ancestry populations
        self.N_anc = self.N_samples[:-1] #numbers of samples in each of ancestral populations
        self.N_targ = self.N_samples[-1] #number of samples in the new population

        #NOTE: each sample refers to the whole individual
        #       sample with id i has [2*i, 2*i+1] haplotype (node) ids 
        self.anc_samples = anc_samples[self.ids]
        
        self.targ_samples = targ_samples
        self.targ_nodes = to_nodes(targ_samples)

        if train_ratio < 0 or train_ratio > 1:
            raise ValueError('incorrect ratio value')
        
        train_samples = []
        ref_samples = []
        train_nodes = []
        ref_nodes = []
        anc_nodes = []
        for i in range(self.anc_pops):
            n = self.N_anc[i]
            n_train = int(n * train_ratio)
            anc = self.anc_samples[i]
            train = np.random.choice(anc, size=n_train, replace=False)
            ref = np.setdiff1d(anc, train)
            train_samples.append(train)
            ref_samples.append(ref)
            train_nodes.append(to_nodes(train))
            ref_nodes.append(to_nodes(ref))
            anc_nodes.append(to_nodes(anc))

        self.train_samples = np.array(train_samples, dtype=object)
        self.ref_samples = np.array(ref_samples, dtype=object)
        self.train_nodes = np.array(train_nodes, dtype=object)
        self.ref_nodes = np.array(ref_nodes, dtype=object)
        self.anc_nodes = np.array(anc_nodes, dtype=object)

        if true_ancestry is not None:
            self.true_ancestry = true_ancestry[:, self.ids] #true ancestry for samples from the new population

        if ts is not None:
            self.ts = ts #tree sequence
            self.seq_len = ts.sequence_length #genome sequence length

    def save_vcf(self):

        if not hasattr(self, "ts") or self.ts is None:
            raise ValueError("tree sequence is not defined")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = f"results/run_{timestamp}"
        Path(self.run_dir).mkdir(parents=True, exist_ok=False)

        self.ts.dump(self.run_dir + "/true_arg.trees")

        vcf_path = self.run_dir + "/sim.vcf.gz"
        with bgzf.open(vcf_path, "wt") as f:
            self.ts.write_vcf(
                f
            )
        subprocess.run(["tabix", "-f", vcf_path], check=True)

        ancestral_states = np.array([site.ancestral_state for site in self.ts.sites()])
        np.save(self.run_dir + "/ancestral_states.npy", ancestral_states)

        zarr_path = vcf_path + ".vcz"
        subprocess.run([
            "python3", "-m", "bio2zarr", "vcf2zarr", "convert",
            "--force", vcf_path, zarr_path
        ], check=True)

        sample_path = self.run_dir + "/sim.samples"
        map_path = self.run_dir + "/sim.map"
        haps_path = self.run_dir + "/sim.haps"

        subprocess.run([
            RELATE_PATH + "/RelateFileFormats",
            "--mode", "ConvertFromVcf",
            "--haps", haps_path,
            "--sample", sample_path,
            "-i", vcf_path.replace(".vcf.gz", "")
        ], check=True)

        with open(map_path, "w") as mapfile:
            mapfile.write('pos COMBINED_rate Genetic_Map\n')
            pos = self.ts.sites_position
            rdist = pos/self.seq_len * 100
            for i in range(pos.shape[0] - 1):
                mapfile.write(f'{pos[i]} {(rdist[i+1] - rdist[i])/(pos[i+1] - pos[i]) * 1e6} {rdist[i]}\n')
            mapfile.write(f'{pos[-1]} {(rdist[-1] - 100)/(pos[-1] - self.seq_len) * 1e6} {rdist[-1]}\n')

        return self.run_dir  

    def anc_cnt(self):
        if not hasattr(self, "ts") or self.ts is None:
            raise ValueError("tree sequence is not defined")
        
        node_pop = {}
        for i in range(self.anc_pops):
            for node in self.anc_nodes[i]:
                node_pop[node] = i
        for node in self.targ_nodes:
            node_pop[node] = -1

        def count_leaf_population(node, tree):
            if node in leaf_population:
                return leaf_population[node]
            cur_pop = np.zeros((self.anc_pops))
            if node in node_pop:
                if node_pop[node] > -1:
                    cur_pop[node_pop[node]] = 1 

                leaf_population[node] = cur_pop
                return cur_pop
            
            for child in tree.children(node):
                child_pop = count_leaf_population(child, tree)
                cur_pop += child_pop

            leaf_population[node] = cur_pop
            return cur_pop

        def count_cousins(node, tree):
            node_pop = count_leaf_population(node, tree)
            cur = node
            cur_pop = count_leaf_population(node, tree)
            while not tree.is_root(cur) and np.array_equal(cur_pop, node_pop):
                cur = tree.parent(cur)
                cur_pop = count_leaf_population(cur, tree)
            return cur_pop - node_pop
        
        self.ancestry_targ = np.zeros((2 * self.N_targ, self.anc_pops, 2 * np.max(self.N_anc) + 1))
        self.ancestry_train = np.zeros((self.anc_pops, self.anc_pops, 2 * np.max(self.N_anc) + 1))

        for tree in self.ts.trees():
            span = tree.span
            span_rel = span / self.seq_len

            leaf_population = dict()
            
            for i in range(self.anc_pops):
                span_rel_rel = span_rel / (2 * self.N_anc[i])
                for node in self.train_nodes[i]:
                    cur_tree_ancestry = count_cousins(node, tree)
                    for j in range(self.anc_pops):
                        self.ancestry_train[i,j,int(cur_tree_ancestry[j])] += span_rel_rel
            
            for i in range(2 * self.N_targ):
                node = self.targ_nodes[i]
                cur_tree_ancestry = count_cousins(node, tree)
                for j in range(self.anc_pops):
                    self.ancestry_targ[i,j,int(cur_tree_ancestry[j])] += span_rel

        return   

    def appr(self, node_idx: int, distance_type: str='KL', allow_unknown_ancestry: bool=False):

        def distance(a, b):
            if distance_type == 'L2':
                return np.sum((a - b) ** 2)
            elif distance_type == 'KL':
                return np.sum(a * np.log((a + eps) / (b + eps)))
            else:
                raise ValueError('unknown distance_type')

        def dist_err(alpha):
            err = 0

            for j in range(self.anc_pops):
                dist = np.tensordot(
                    alpha,
                    self.ancestry_train[:, j, :],
                    axes=1
                )
                err += distance(self.ancestry_targ[node_idx, j], dist)
            
            return err
        
        if not (hasattr(self, "ancestry_targ") and hasattr(self, "ancestry_train")) or self.ancestry_targ is None or self.ancestry_train is None:
            raise ValueError("ancestry is not defined yet, first use anc_cnt() function")

        if allow_unknown_ancestry:
            constraints = [{"type": "ineq", "fun": lambda x: 1 - np.sum(x)}]
        else:
            constraints = [{"type": "eq", "fun": lambda x: 1 - np.sum(x)}]

        alpha_res = minimize(
            dist_err,
            np.ones(self.anc_pops) / self.anc_pops,
            bounds = [(0, 1)] * self.anc_pops,
            constraints = constraints,
            method="SLSQP"
        )

        if not alpha_res.success:
            raise RuntimeError(alpha_res.message)
        
        alpha = alpha_res.x
        return alpha, dist_err(alpha)
        
    def targ_appr(self, distance_type: str='KL', allow_unknown_ancestry: bool=False):
        alpha = []
        d_err = []
        for node_idx in range(2 * self.N_targ):
            a, e = self.appr(node_idx, distance_type=distance_type, allow_unknown_ancestry=allow_unknown_ancestry)
            alpha.append(a)
            d_err.append(e)

        return np.array(alpha), np.array(d_err)
    
    def targ_appr_res(self, pop_id:int=0, distance_type: str='KL', allow_unknown_ancestry: bool=False):
        alpha, _ = self.targ_appr(distance_type=distance_type, allow_unknown_ancestry=allow_unknown_ancestry)
        t_a = self.true_ancestry[:,pop_id]
        a_a = alpha[:,pop_id]
        _, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].set_xlabel('actual proportion')
        ax[0].set_ylabel('proportion approximation')
        ax[0].plot([0,1], [0,1], alpha=0.3)
        ax[0].scatter(t_a, a_a)
        ax[0].set_title(f'proportion of ancestry from the population number {self.ids[pop_id] + 1}\n' + 
                        f'MSE = {np.mean((t_a - a_a)**2)}\n' +
                        f'approximation with {distance_type} type of distribution distance')
        
        ax[1].boxplot(a_a - t_a)
        ax[1].axhline(y=0, color='grey', alpha=0.5) 
        plt.show()
        return


class TS_sim(TS_model):
    def __init__(self, 
                t_anc: float=500, t_new: float=5, 
                admix_ratio: None|np.ndarray|list=None, rho=1.2e-8, mu=1.25e-8,
                anc_pops: None|int=None, N_samples: None|np.ndarray|list=None,
                size_com: int=1e3, size_anc: None|np.ndarray|list=None, size_new: int=1e3, 
                seq_len: int|None=None, train_ratio: float=0.4):

        if N_samples is None:
            if anc_pops is None:
                anc_pops = 2
            N_samples = np.array([50] * (anc_pops + 1))
        if anc_pops is None:
            anc_pops = len(N_samples) - 1
        self.anc_pops = anc_pops

        self.mu = mu
        self.N = np.sum(N_samples)
        self.N_samples = N_samples

        if size_anc is None:
            size_anc = np.array([1e3] * anc_pops)
        if admix_ratio is None:
            admix_ratio = np.array([1 / anc_pops] * anc_pops)
        if seq_len is None:
            seq_len = int(1 / rho)
        
        self.dem_sim(t_anc, t_new, admix_ratio, size_com, size_anc, size_new)
        ts, anc_samples, targ_samples = self.anc_sim(N_samples, rho, mu, seq_len)
        true_ancestry = self.true_anc()

        super().__init__(anc_samples, targ_samples, ts, train_ratio, true_ancestry)

    def dem_sim(self, t_anc: float, t_new: float, admix_ratio: np.ndarray | list, 
                size_com: int, size_anc: np.ndarray | list, size_new: int):

        if size_anc is None:
            size_anc = [size_new] * self.anc_pops

        demography = msprime.Demography()
        demography.add_population(name='com', initial_size=size_com)
        anc_list = [f'anc_{i}' for i in range(self.anc_pops)]
        for i in range(self.anc_pops):
            demography.add_population(name=anc_list[i], initial_size=size_anc[i])
        demography.add_population(name='new', initial_size=size_new)

        demography.add_admixture(time=t_new, derived='new', ancestral=anc_list, proportions=admix_ratio)
        demography.add_census(t_new + 1)
        demography.add_population_split(time=t_new + t_anc, derived=anc_list, ancestral='com')
        self.demo = demography
        return demography
    
    def dem_pic(self):
        pic = msprime.Demography.to_demes(self.demo)
        demesdraw.tubes(pic)
        return
    
    def anc_sim(self, N_samples: np.ndarray | list, rho: float, mu: float, seq_len: int):
        if seq_len is None:
            seq_len = int(1 / rho)
        self.seq_len = seq_len
        samples=dict()
        samples['com'] = 0
        for i in range(self.anc_pops):
            samples[f'anc_{i}'] = N_samples[i]
        samples['new'] = N_samples[-1]

        ts = msprime.sim_ancestry(samples=samples, demography=self.demo, sequence_length=seq_len, recombination_rate=rho, ploidy=2)
        ts = msprime.sim_mutations(tree_sequence=ts, rate=mu)
        self.ts = ts

        anc_samples = []
        for i in range(len(N_samples) - 1):
            anc_samples.append(np.arange(np.sum(N_samples[:i]), np.sum(N_samples[:i+1])))
        anc_samples = np.array(anc_samples, dtype=object)
        
        targ_samples = np.arange(np.sum(N_samples[:-1]), np.sum(N_samples))

        self.anc_samples = anc_samples
        self.targ_samples = targ_samples
        return ts, anc_samples, targ_samples

    def true_anc(self):

        def count_local_ancestry(node, tree):
            pop = tree.population(node)
            if pop == self.anc_pops + 1:
                return count_local_ancestry(tree.parent(node), tree)
            if pop == 0 or tree.is_root(node):
                return 'com'
            return pop - 1
        
        true_ancestry = np.zeros((2 * self.N_samples[-1], self.anc_pops))
        for tree in self.ts.trees():
            span_rel = tree.span / self.seq_len
            for i in range(self.N_samples[-1]):
                sample = self.targ_samples[i]
                for h in range(2):
                    node = 2 * sample + h
                    pop = count_local_ancestry(node, tree)
                    if pop != 'com':
                        true_ancestry[2 * i + h, pop] += span_rel
        self.true_ancestry = true_ancestry
        return true_ancestry

class TS_tsinfer(TS_model):
    def __init__(self, ts_sim:TS_model, run_dir: str, 
                 vcf_path:str|None=None, ancestral_states_path:str|None=None, 
                 train_ratio: float=0.4):

        if vcf_path is None:
            vcf_path = run_dir + "/sim.vcf.gz"
        if ancestral_states_path is None:
            ancestral_states_path = run_dir + "/ancestral_states.npy"
        zarr_path = vcf_path + ".vcz"

        for f in [zarr_path, ancestral_states_path]:
            if not Path(f).exists():
                raise FileNotFoundError()
            
        ancestral_states = np.load(ancestral_states_path)
        vdata = tsinfer.VariantData(zarr_path, ancestral_states)
        ts = tsinfer.infer(vdata, progress_monitor=False)

        super().__init__(ts_sim.anc_samples, ts_sim.targ_samples, 
                         ts, train_ratio)
        self.true_ancestry = ts_sim.true_ancestry

class TS_relate(TS_model):
    def __init__(self, ts_sim:TS_model, run_dir: str, 
                 haps_path: str|None=None, sample_path: str|None=None, map_path: str|None=None,
                 train_ratio: float=0.4):
        if haps_path is None:
            haps_path = "sim.haps"
        if sample_path is None:
            sample_path = "sim.samples"
        if map_path is None:
            map_path = "sim.map"

        subprocess.run([
            RELATE_PATH + "/Relate",
            "--mode", "All",
            "--haps", haps_path,
            "--sample", sample_path,
            "--map", map_path,
            "-m", str(ts_sim.mu),
            "-N", str(ts_sim.N), 
            "--output", "relate_arg"
        ], check=True, cwd=run_dir)

        subprocess.run([
            RELATE_PATH + "/RelateFileFormats",
            "--mode", "ConvertToTreeSequence",
            "-i", "relate_arg",
            "-o", "relate_arg"
        ], check=True, cwd=run_dir)

        ts = tskit.load(run_dir + "/relate_arg.trees")

        super().__init__(ts_sim.anc_samples, ts_sim.targ_samples, 
                         ts, train_ratio)
        self.true_ancestry = ts_sim.true_ancestry
        