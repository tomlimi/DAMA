import numpy as np
import matplotlib.pyplot as plt
import scipy
import os

from sklearn.linear_model import LinearRegression


# Functions for locating the subject in the sentence
def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]

def find_token_range(tokenizer, string, substring):
    toks = decode_tokens(tokenizer, tokenizer.encode(string))
    whole_string = "".join(toks)
    substring = "".join(decode_tokens(tokenizer, tokenizer.encode(substring)))
    # whole_string = tokenizer.decode(token_array)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)

# Function aligning the scores with the positions in subject after subject
class Avg:
    def __init__(self):
        self.d = []

    def add(self, v):
        self.d.append(v[None])

    def add_all(self, vv):
        self.d.append(vv)

    def avg(self):
        return np.concatenate(self.d).mean(axis=0)

    def std(self):
        return np.concatenate(self.d).std(axis=0)

    def size(self):
        return sum(datum.shape[0] for datum in self.d)


def read_data(data, tokenizer, kind=None, in_prompt=None):
    (
        avg_fs,
        avg_es,
        avg_ls,
        avg_fp,
        avg_ep,
        avg_lp,
        avg_m,
        avg_mlp,
        avg_mls,
        avg_low,
        avg_high
    ) = [Avg() for _ in range(11)]
    for data_row in data:
        if in_prompt is not None and data_row["prompt"].find(in_prompt) == -1:
            continue

        subject_range = find_token_range(tokenizer, data_row["prompt"], data_row["subject"])

        scores = np.array(data_row[kind]["scores"])
        first_s, first_p = subject_range
        last_s = first_p - 1
        last_p = len(scores) - 1
        # original prediction
        avg_high.add(np.array(data_row[kind]["high_score"]))
        # prediction after subject is corrupted
        avg_low.add(np.array(data_row[kind]["low_score"]))
        avg_m.add(scores.max(axis=(0,1)))
        # some maximum computations
        avg_mls.add(scores[last_s].max(axis=0))
        avg_mlp.add(scores[last_p].max(axis=0))
        # First subject middle, last subjet.
        avg_fs.add(scores[first_s])
        avg_es.add_all(scores[first_s + 1 : last_s])
        avg_ls.add(scores[last_s])
        # First after, middle after, last after
        avg_fp.add(scores[first_p])
        avg_ep.add_all(scores[first_p + 1 : last_p])
        avg_lp.add(scores[last_p])

    result = np.stack(
        [
            avg_fs.avg(),
            avg_es.avg(),
            avg_ls.avg(),
            avg_fp.avg(),
            avg_ep.avg(),
            avg_lp.avg(),
        ]
    )
    result_std = np.stack(
        [
            avg_fs.std(),
            avg_es.std(),
            avg_ls.std(),
            avg_fp.std(),
            avg_ep.std(),
            avg_lp.std(),
        ]
    )
    print("Type of representation:", kind)
    print("Average Total Effect", avg_high.avg() - avg_low.avg())
    print(
        "Best average indirect effect on last subject",
        avg_ls.avg().max(axis=0) - avg_low.avg(),
    )
    print(
        "Best average indirect effect on last token", avg_lp.avg().max(axis=0) - avg_low.avg()
    )
    print("Average best-fixed score", avg_m.avg())
    print("Average best-fixed on last subject token score", avg_mls.avg())
    print("Average best-fixed on last word score", avg_mlp.avg())
    print("Argmax at last subject token", np.argmax(avg_ls.avg(), axis=0))
    print("Max at last subject token", np.max(avg_ls.avg(), axis=0))
    print("Argmax at last prompt token", np.argmax(avg_lp.avg(), axis=0))
    print("Max at last prompt token", np.max(avg_lp.avg(), axis=0))
    return dict(
        high_score=avg_high.avg() ,low_score=avg_low.avg(), result=result, result_std=result_std, size=avg_fs.size()
    )

def aggregate_token_scores(data_row, tokenizer, kind=None):
    subject_range = find_token_range(tokenizer, data_row["prompt"], data_row["subject"])
    
    scores = np.array(data_row[kind]["scores"])
    first_s, first_p = subject_range
    last_s = first_p - 1
    last_p = len(scores) - 1
    
    result = np.stack([
        scores[first_s],
        scores[first_s + 1: last_s].mean(axis=0) if first_s + 1 < last_s else np.zeros_like(scores[first_s]),
        scores[last_s],
        scores[first_p],
        scores[first_p + 1: last_p].mean(axis=0) if first_p + 1 < last_p else np.zeros_like(scores[first_p]),
        scores[last_p]
        ],
        axis=0
    )
    
    return result


# function for average causal tracing
def plot_array(
    all_differences,
    kind=None,
    savepdf=None,
    title=None,
    all_low_scores=None,
    all_high_scores=None,
):
    if all_low_scores is None:
        all_low_scores = all_differences.min()
    if all_high_scores is None:
        all_high_scores = all_differences.max()

    labels = [
        "First subject token",
        "Middle subject tokens",
        "Last subject token",
        "First subsequent token",
        "Further tokens",
        "Last token",
    ]

    all_low = np.min(all_low_scores)
    all_high = np.max(all_differences)
    fig, axes = plt.subplots(1,4, figsize=(12, 2), dpi=200)
    all_answers = ["she", "he", "they", "he - she"]
    for differences, low_score, high_score, answer, ax, ax_id in zip(np.split(all_differences, 4,-1),
                                                      all_low_scores, all_high_scores,
                                                      all_answers, axes, range(4)):
        differences = differences.squeeze()
        h = ax.pcolor(
            differences,
            cmap={"null": "Purples", "mlp": "Greens", "attn": "Reds"}[kind],
            vmin=low_score,
            vmax=differences.max(),#high_score
        )
        if title and ax_id == 1:
            ax.set_title(title)
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if ax_id == 0:
            ax.set_yticklabels(labels)
        else:
            ax.set_yticks([])
        if ax_id == 1:
            if not kind:
                ax.set_xlabel(f"single restored layer within llama")
            else:
                ax.set_xlabel(f"center of interval restored {kind} layers")

        cb = plt.colorbar(h)
        # The following should be cb.ax.set_xlabel(answer), but this is broken in matplotlib 3.5.1.
        if answer:
            cb.ax.set_title(str(answer).strip(), y=-0.16, fontsize=10)


# functions for factual_stereotypical traces

TOKEN_POSITION_MAP = (
        "First subject token",
        "Middle subject tokens",
        "Last subject token",
        "First subsequent token",
        "Further tokens",
        "Last token"
)

def plot_effect_against_bias(data, professions, layer=25, kind="null", token_positions=[0,2,3,5], in_prompt=None, log=False):
    print(f"Plotting {kind} at layer {layer}")
    fig, axs = plt.subplots(1, len(token_positions), figsize=(5*len(token_positions), 5))
    for i, token_position in enumerate(token_positions):
        ax = axs[i]

        ie_scores = []
        f_scores = []
        s_scores = []

        for data_row in data:
            if in_prompt is not None and data_row['prompt'].find(in_prompt) == -1:
                continue
                
            # probabilty of he - she
            if log:
                ie_score = np.log(1e-4  +data_row[kind]["scores_aggregated"][token_position, layer, 1]) \
                           - np.log(1e-4 + data_row[kind]["scores_aggregated"][token_position, layer, 0])
            else:
                ie_score = data_row[kind]["scores_aggregated"][token_position,layer, 1] \
                           - data_row[kind]["scores_aggregated"][token_position,layer, 0]
            subject = data_row['subject']
            subject = subject[4:].replace(" ", "_")
            f_score = professions[subject]['factual']
            s_score = professions[subject]['stereotypical']

            ie_scores.append(ie_score)
            f_scores.append(f_score)
            s_scores.append(s_score)

        # plot scatterplot with two series with seaborn
        # sns.regplot(x="gender", y="ie", hue='type', data=pd.DataFrame({"gender": f_scores + s_scores, "ie": list(ie_scores) + list(ie_scores),
        #                                                    "type": ["factual"] * len(f_scores) + ["stereotypical"] * len(s_scores)}),
        #            scatter_kws={"alpha": 0.3}, ax=ax)

        f_scorr = scipy.stats.spearmanr(f_scores, ie_scores)[0]
        s_scorr = scipy.stats.spearmanr(s_scores, ie_scores)[0]
        ax.scatter(f_scores, ie_scores, alpha=0.3, label=f"factual r={f_scorr:.2f}")
        ax.scatter(s_scores, ie_scores, alpha=0.3, label=f"stereotypical r={s_scorr:.2f}")
        ax.legend()

        ax.set_title(TOKEN_POSITION_MAP[token_position])
        if i == 0:
            ax.set_ylabel("IE: he - she")
        ax.set_xlabel("gender score")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-.6, .6)
    plt.show()

def plot_correlations(data, professions, kind="null", in_prompt=None, log=False):

    ie_scores = []
    f_scores = []
    s_scores = []
    
    h_scores = []

    for data_row in data:
        if in_prompt is not None and data_row['prompt'].find(in_prompt) == -1:
            continue
        # probabilty of he - she
        if log:
            ie_score  = np.log(1e-4 + data_row[kind]["scores_aggregated"][:,:, 1])  - np.log(1e-4 + data_row[kind]["scores_aggregated"][:,:, 0])
            h_score = np.log(1e-4 + data_row[kind]["high_score"][1]) - np.log(1e-4 + data_row[kind]["high_score"][0])
        else:
            ie_score = data_row[kind]["scores_aggregated"][:,:, 1] - data_row[kind]["scores_aggregated"][:,:, 0]
            h_score = data_row[kind]["high_score"][1] - data_row[kind]["high_score"][0]
        subject = data_row['subject']
        subject = subject[4:].replace(" ", "_")

        f_score = professions[subject]['factual']
        s_score = professions[subject]['stereotypical']



        ie_scores.append(ie_score)
        f_scores.append(f_score)
        s_scores.append(s_score)
        h_scores.append(h_score)
    #print(ie_scores)
    # compute spearmna correlation for each n_layers, n_token_positions combination in ie_scores
    # we want to compute the correlation for each layer and token position


    ie_scores = np.stack(ie_scores)
    f_sccorrs = np.zeros((ie_scores.shape[1], ie_scores.shape[2]))
    s_sccorrs = np.zeros((ie_scores.shape[1], ie_scores.shape[2]))
    h_sccorrs = np.zeros((ie_scores.shape[1], ie_scores.shape[2]))
    for token_position in range(ie_scores.shape[1]):
        for layer in range(ie_scores.shape[2]):
            # compute correlation
            f_sccorrs[token_position, layer] = np.nan_to_num(scipy.stats.spearmanr(f_scores, ie_scores[:, token_position, layer])[0])
            s_sccorrs[token_position, layer] = np.nan_to_num(scipy.stats.spearmanr(s_scores, ie_scores[:, token_position, layer])[0])
            h_sccorrs[token_position, layer] = np.nan_to_num(scipy.stats.spearmanr(h_scores, ie_scores[:, token_position, layer])[0])

    
    m_tp, m_l = np.unravel_index(np.argmax(f_sccorrs - s_sccorrs, axis=None), f_sccorrs.shape)
    print(f"Maximal difference in Spearman correlation F > S at layer {m_l} at {TOKEN_POSITION_MAP[m_tp]}")
    print(f"Stereotypical r={s_sccorrs[m_tp, m_l]:2f}")
    print(f"Factual r={f_sccorrs[m_tp, m_l]:2f}")

    m2_tp, m2_l = np.unravel_index(np.argmax(s_sccorrs - f_sccorrs, axis=None), f_sccorrs.shape)
    print(f"Maximal difference in Spearman correlation S > F  at layer {m2_l} at {TOKEN_POSITION_MAP[m2_tp]}")
    print(f"Stereotypical r={s_sccorrs[m2_tp, m2_l]:2f}")
    print(f"Factual r={f_sccorrs[m2_tp, m2_l]:2f}")

    # plot correlations
    print(" Total effect correlations")
    print("stereotypical: ", scipy.stats.spearmanr(s_scores, h_scores)[0])
    print("factual: ", scipy.stats.spearmanr(f_scores, h_scores)[0])
    
    fig, axes = plt.subplots(1,4, figsize=(10, 2), dpi=200)
    h = axes[0].pcolor(
        f_sccorrs,
        # cmap={"null": "Purples", "mlp": "Greens", "attn": "Reds"}[kind],
        # vmin=min(f_sccorrs.min(), s_sccorrs.min()),
        # vmax=max(f_sccorrs.max(), s_sccorrs.max())
        cmap='RdYlGn',
        vmin=-max(f_sccorrs.max(), s_sccorrs.max(), h_sccorrs.max()),
        vmax=max(f_sccorrs.max(), s_sccorrs.max(), h_sccorrs.max())
        )
    axes[0].set_title(f"Factual")
    axes[0].set_yticklabels(TOKEN_POSITION_MAP)
    axes[0].set_ylabel("token position")
    h= axes[1].pcolor(
        s_sccorrs,
        # cmap={"null": "Purples", "mlp": "Greens", "attn": "Reds"}[kind],
        # vmin=min(f_sccorrs.min(), s_sccorrs.min()),
        # vmax=max(f_sccorrs.max(), s_sccorrs.max())
        cmap='RdYlGn',
        vmin=-max(f_sccorrs.max(), s_sccorrs.max(), h_sccorrs.max()),
        vmax=max(f_sccorrs.max(), s_sccorrs.max(), h_sccorrs.max())
        )
    axes[1].set_title(f"Stereotypical")
    axes[1].set_yticklabels([])
    h = axes[2].pcolor(
        h_sccorrs,
        cmap='RdYlGn',
        vmin=-max(f_sccorrs.max(), s_sccorrs.max(), h_sccorrs.max()),
        vmax=max(f_sccorrs.max(), s_sccorrs.max(), h_sccorrs.max())
    )
    axes[2].set_title(f"Total effect")
    axes[2].set_yticklabels([])
    
    h2 = axes[3].pcolor(
        f_sccorrs - s_sccorrs,
        cmap='RdYlGn',
        vmin=-max((f_sccorrs - s_sccorrs).max(), (s_sccorrs - f_sccorrs).max()),
        vmax=max((f_sccorrs - s_sccorrs).max(), (s_sccorrs - f_sccorrs).max())
        )
    axes[3].set_title(f"Diff F - S")
    axes[3].set_yticklabels([])


    for ax in axes:
        ax.set_xlabel("layer")

        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(f_sccorrs))])
        ax.set_xticks([0.5 + i for i in range(0, f_sccorrs.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, f_sccorrs.shape[1] - 6, 5)))
    fig.suptitle(f"Spearman correlations {kind} IE and:", y=1.1)
    plt.colorbar(h, ax=axes[:3].ravel().tolist())
    plt.colorbar(h2, ax=axes[3])
    plt.show()


def plot_joint_linear_coefficients(data, professions, kind="null", in_prompt=None, log=False, savedir=None,
                                   param=None):
    ie_scores = []
    f_scores = []
    s_scores = []
    
    h_scores = []
    
    for data_row in data:
        if in_prompt is not None and data_row['prompt'].find(in_prompt) == -1:
            continue
        # probabilty of he - she
        if log:
            ie_score = np.log(1e-4 + data_row[kind]["scores_aggregated"][:, :, 1]) - np.log(
                1e-4 + data_row[kind]["scores_aggregated"][:, :, 0])
            h_score = np.log(1e-4 + data_row[kind]["high_score"][1]) - np.log(1e-4 + data_row[kind]["high_score"][0])
        else:
            ie_score = data_row[kind]["scores_aggregated"][:, :, 1] - data_row[kind]["scores_aggregated"][:, :, 0]
            h_score = data_row[kind]["high_score"][1] - data_row[kind]["high_score"][0]
        subject = data_row['subject']
        subject = subject[4:].replace(" ", "_")
        
        f_score = professions[subject]['factual']
        s_score = professions[subject]['stereotypical']
        
        ie_scores.append(ie_score)
        f_scores.append(f_score)
        s_scores.append(s_score)
        h_scores.append(h_score)
    # print(ie_scores)
    # compute spearmna correlation for each n_layers, n_token_positions combination in ie_scores
    # we want to compute the correlation for each layer and token position
    
    ie_scores = np.stack(ie_scores)
    sf_scores = np.stack([f_scores, s_scores], axis=1)
    
    f_acoeffs = np.zeros((ie_scores.shape[1], ie_scores.shape[2]))
    s_acoeffs = np.zeros((ie_scores.shape[1], ie_scores.shape[2]))
    bcoeffs = np.zeros((ie_scores.shape[1], ie_scores.shape[2]))
    r2s = np.zeros((ie_scores.shape[1], ie_scores.shape[2]))
    
    for token_position in range(ie_scores.shape[1]):
        for layer in range(ie_scores.shape[2]):
            # fit linear regression and save coefficients
            reg = LinearRegression().fit(sf_scores, ie_scores[:, token_position, layer])
            f_acoeffs[token_position, layer], s_acoeffs[token_position, layer], bcoeffs[token_position, layer] = reg.coef_[0], reg.coef_[1], reg.intercept_
            r2s[token_position, layer] = reg.score(sf_scores, ie_scores[:, token_position, layer])
            
            
    # fit two-dimensional linear model
    sf_scores = np.stack([f_scores, s_scores], axis=1)
    
    h_scores = np.array(h_scores)
    reg  = LinearRegression().fit(sf_scores, h_scores)
    
    f_acoeff = reg.coef_[0]
    s_acoeff = reg.coef_[1]
    bcoeff = reg.intercept_
    r2 = reg.score(sf_scores, h_scores.reshape((ie_scores.shape[0], -1)))
    
    print(f"TE coefficients: stereotypical a_s={s_acoeff}, factual a_f={f_acoeff}, intercept b={bcoeff}, r2={r2}")
    sf_pearson = scipy.stats.pearsonr(f_scores, s_scores)
    print("Factual vs stereotypical correlation: ", sf_pearson)
    
    fig, axes = plt.subplots(1, 4, figsize=(10, 2), dpi=200)
    axes = axes.ravel()
    
    h = axes[0].pcolor(
        f_acoeffs,
        cmap='RdYlGn',
        vmin=-max(f_acoeffs.max(), s_acoeffs.max()),
        vmax=max(f_acoeffs.max(), s_acoeffs.max())
    )
    axes[0].set_title(f"Factual")
    axes[0].set_yticklabels(TOKEN_POSITION_MAP)
    
    h = axes[1].pcolor(
        s_acoeffs,
        cmap='RdYlGn',
        vmin=-max(f_acoeffs.max(), s_acoeffs.max()),
        vmax=max(f_acoeffs.max(), s_acoeffs.max())
    )
    axes[1].set_title(f"Stereotypical")
    axes[1].set_yticklabels([])
    h1 = axes[2].pcolor(
        bcoeffs,
        cmap='RdYlGn',
        vmin=-bcoeffs.max(),
        vmax=bcoeffs.max()
    )
    axes[2].set_title(f"Intercept")
    axes[2].set_yticklabels([])
    
    h2 = axes[3].pcolor(
        r2s,
        cmap='RdYlGn',
        vmin=-r2s.max(),
        vmax=r2s.max()
    )
    axes[3].set_title(r"$R^2$")
    axes[3].set_yticklabels([])
    
    plt.colorbar(h, ax=axes[:2].ravel().tolist())
    plt.colorbar(h1, ax=axes[2])
    plt.colorbar(h2, ax=axes[3])
    
    for ax in axes:
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(bcoeffs))])
        ax.set_xticks([0.5 + i for i in range(0, bcoeffs.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, bcoeffs.shape[1] - 6, 5)))
    #fig.suptitle(r"Linear coefficient $y = a_s \cdot x_s + a_f \cdot x_f + b$" + f" for {kind} IE", y=1.2)
    if savedir is not None:
        save_path = os.path.join(savedir, f"{param}B_corrcoeff_{kind}_IE.pdf")
        # plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        
    plt.show()

def plot_linear_coefficients(data, professions, kind="null", in_prompt=None, log=False):
    ie_scores = []
    f_scores = []
    s_scores = []
    
    h_scores = []
    
    for data_row in data:
        if in_prompt is not None and data_row['prompt'].find(in_prompt) == -1:
            continue
        # probabilty of he - she
        if log:
            ie_score  = np.log(1e-4 + data_row[kind]["scores_aggregated"][:,:, 1])  - np.log(1e-4 + data_row[kind]["scores_aggregated"][:,:, 0])
            h_score = np.log(1e-4 + data_row[kind]["high_score"][1]) - np.log(1e-4 + data_row[kind]["high_score"][0])
        else:
            ie_score = data_row[kind]["scores_aggregated"][:,:, 1] - data_row[kind]["scores_aggregated"][:,:, 0]
            h_score = data_row[kind]["high_score"][1] - data_row[kind]["high_score"][0]
        subject = data_row['subject']
        subject = subject[4:].replace(" ", "_")
        
        f_score = professions[subject]['factual']
        s_score = professions[subject]['stereotypical']


        ie_scores.append(ie_score)
        f_scores.append(f_score)
        s_scores.append(s_score)
        h_scores.append(h_score)
    # print(ie_scores)
    # compute spearmna correlation for each n_layers, n_token_positions combination in ie_scores
    # we want to compute the correlation for each layer and token position
    
    ie_scores = np.stack(ie_scores)
    f_acoeff = np.zeros((ie_scores.shape[1], ie_scores.shape[2]))
    s_acoeff = np.zeros((ie_scores.shape[1], ie_scores.shape[2]))
    h_acoeff= np.zeros((ie_scores.shape[1], ie_scores.shape[2]))
    
    f_bcoeff = np.zeros((ie_scores.shape[1], ie_scores.shape[2]))
    s_bcoeff = np.zeros((ie_scores.shape[1], ie_scores.shape[2]))
    h_bcoeff = np.zeros((ie_scores.shape[1], ie_scores.shape[2]))
    
    
    # compute linear regression coefficients
    f_rfit = np.zeros((ie_scores.shape[1], ie_scores.shape[2]))
    s_rfit = np.zeros((ie_scores.shape[1], ie_scores.shape[2]))
    h_rfit = np.zeros((ie_scores.shape[1], ie_scores.shape[2]))
    
    
    for token_position in range(ie_scores.shape[1]):
        for layer in range(ie_scores.shape[2]):
            # fit linear regression and save coefficients
            f_acoeff[token_position, layer], f_bcoeff[token_position, layer], f_rfit[token_position, layer] , _, _ = scipy.stats.linregress(f_scores, ie_scores[:, token_position, layer])
            s_acoeff[token_position, layer], s_bcoeff[token_position, layer], s_rfit[token_position, layer] , _, _ = scipy.stats.linregress(s_scores, ie_scores[:, token_position, layer])
            h_acoeff[token_position, layer], h_bcoeff[token_position, layer], h_rfit[token_position, layer] , _, _ = scipy.stats.linregress(h_scores, ie_scores[:, token_position, layer])

    f_rfit = f_rfit**2
    s_rfit = s_rfit**2
    h_rfit = h_rfit**2

    
    # plot correlations
    print(" Total effect coefficients")
    sh_acoef, sh_bcoef, sh_rfit, _, _ = scipy.stats.linregress(s_scores, h_scores)
    fh_acoef, fh_bcoef, fh_rfit, _, _ = scipy.stats.linregress(f_scores, h_scores)
    print("stereotypical: ", np.polyfit(s_scores, h_scores, 1))
    print(f"stereotypical a={sh_acoef}, b={sh_bcoef}, r2={sh_rfit**2}")
    print("factual: ", np.polyfit(f_scores, h_scores, 1))
    print(f"factual a={fh_acoef}, b={fh_bcoef}, r2={fh_rfit**2}")
    
    fig, axes = plt.subplots(3, 4, figsize=(10, 6), dpi=200)
    axes = axes.ravel()
    
    h = axes[0].pcolor(
        f_acoeff,
        cmap='RdYlGn',
        vmin=-max(f_acoeff.max(), s_acoeff.max(), h_acoeff.max()),
        vmax=max(f_acoeff.max(), s_acoeff.max(), h_acoeff.max())
    )
    axes[0].set_title(f"Factual")
    axes[0].set_yticklabels(TOKEN_POSITION_MAP)
    axes[0].set_ylabel("a")
    
    h = axes[1].pcolor(
        s_acoeff,
        cmap='RdYlGn',
        vmin=-max(f_acoeff.max(), s_acoeff.max(), h_acoeff.max()),
        vmax=max(f_acoeff.max(), s_acoeff.max(), h_acoeff.max())
    )
    axes[1].set_title(f"Stereotypical")
    axes[1].set_yticklabels([])
    h = axes[2].pcolor(
        h_acoeff,
        cmap='RdYlGn',
        vmin=-max(f_acoeff.max(), s_acoeff.max(), h_acoeff.max()),
        vmax=max(f_acoeff.max(), s_acoeff.max(), h_acoeff.max())
    )
    axes[2].set_title(f"Total effect")
    axes[2].set_yticklabels([])
    
    h2 = axes[3].pcolor(
        f_acoeff - s_acoeff,
        cmap='RdYlGn',
        vmin=-max((f_acoeff - s_acoeff).max(), (s_acoeff - f_acoeff).max()),
        vmax=max((f_acoeff - s_acoeff).max(), (s_acoeff - f_acoeff).max())
    )
    axes[3].set_title(f"Diff F - S")
    axes[3].set_yticklabels([])
    

    
    plt.colorbar(h, ax=axes[:3].ravel().tolist())
    plt.colorbar(h2, ax=axes[3])
    
    h = axes[4].pcolor(
        f_bcoeff,
        cmap='RdYlGn',
        vmin=-max(f_bcoeff.max(), s_bcoeff.max(), h_bcoeff.max()),
        vmax=max(f_bcoeff.max(), s_bcoeff.max(), h_bcoeff.max())
    )
    

    axes[4].set_yticklabels(TOKEN_POSITION_MAP)
    axes[4].set_ylabel("b")
    h = axes[5].pcolor(
        s_bcoeff,
        cmap='RdYlGn',
        vmin=-max(f_bcoeff.max(), s_bcoeff.max(), h_bcoeff.max()),
        vmax=max(f_bcoeff.max(), s_bcoeff.max(), h_bcoeff.max())
    )

    axes[5].set_yticklabels([])
    h = axes[6].pcolor(
        h_bcoeff,
        cmap='RdYlGn',
        vmin=-max(f_bcoeff.max(), s_bcoeff.max(), h_bcoeff.max()),
        vmax=max(f_bcoeff.max(), s_bcoeff.max(), h_bcoeff.max())
    )

    axes[6].set_yticklabels([])
    
    h2 = axes[7].pcolor(
        f_bcoeff - s_bcoeff,
        cmap='RdYlGn',
        vmin=-max((f_bcoeff - s_bcoeff).max(), (s_bcoeff - f_bcoeff).max()),
        vmax=max((f_bcoeff - s_bcoeff).max(), (s_bcoeff - f_bcoeff).max())
    )

    axes[7].set_yticklabels([])
    

        
    plt.colorbar(h, ax=axes[4:7].ravel().tolist())
    plt.colorbar(h2, ax=axes[7])
    
    h = axes[8].pcolor(
        s_rfit,
        cmap='RdYlGn',
        vmin=-max(f_rfit.max(), s_rfit.max(), h_rfit.max()),
        vmax=max(f_rfit.max(), s_rfit.max(), h_rfit.max())
    )
    axes[8].set_ylabel(r"$R^2$")
    axes[8].set_yticklabels(TOKEN_POSITION_MAP)
    
    h = axes[9].pcolor(
        f_rfit,
        cmap='RdYlGn',
        vmin=-max(f_rfit.max(), s_rfit.max(), h_rfit.max()),
        vmax=max(f_rfit.max(), s_rfit.max(), h_rfit.max())
    )

    axes[9].set_yticklabels([])
    
    h = axes[10].pcolor(
        h_rfit,
        cmap='RdYlGn',
        vmin=-max(f_rfit.max(), s_rfit.max(), h_rfit.max()),
        vmax=max(f_rfit.max(), s_rfit.max(), h_rfit.max())
    )
    
    axes[10].set_yticklabels([])
    
    h2 = axes[11].pcolor(
        f_rfit - s_rfit,
        cmap='RdYlGn',
        vmin=-max((f_rfit - s_rfit).max(), (s_rfit - f_rfit).max()),
        vmax=max((f_rfit - s_rfit).max(), (s_rfit - f_rfit).max())
    )
    
    axes[11].set_yticklabels([])
    

    plt.colorbar(h, ax=axes[8:11].ravel().tolist())
    plt.colorbar(h2, ax=axes[11])
    
    for ax in axes:
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(f_bcoeff))])
        ax.set_xticks([0.5 + i for i in range(0, f_bcoeff.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, f_bcoeff.shape[1] - 6, 5)))
    fig.suptitle(f"Linear coefficient y = x*a + b : {kind} IE and:", y=1.1)

    plt.show()