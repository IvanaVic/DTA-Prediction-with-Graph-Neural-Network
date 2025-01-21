import os
import numpy as np
import Bio.PDB
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import traceback
import ast
import networkx as nx

pro_res_table = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
                   'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'SEC']
amino_acids_set = set(pro_res_table)

pro_res_aliphatic_table = ['ALA', 'VAL', 'LEU', 'ILE', 'MET']
pro_res_aromatic_table = ['PHE', 'TYR', 'TRP']
pro_res_polar_neutral_table = ['SER', 'THR', 'ASN', 'GLN', 'CYS']
pro_res_acidic_charged_table = ['ASP', 'GLU']
pro_res_basic_charged_table = ['ARG', 'LYS', 'HIS']

res_weight_table = {'ALA': 71.08, 'CYS': 103.15, 'ASP': 115.09, 'GLU': 129.12, 'PHE': 147.18, 'GLY': 57.05, 'HIS': 137.14,
                    'ILE': 113.16, 'LYS': 128.18, 'LEU': 113.16, 'MET': 131.20, 'ASN': 114.11, 'PRO': 97.12, 'GLN': 128.13,
                    'ARG': 156.19, 'SER': 87.08, 'THR': 101.11, 'VAL': 99.13, 'TRP': 186.22, 'TYR': 163.18}

res_pka_table = {'ALA': 2.34, 'CYS': 1.96, 'ASP': 1.88, 'GLU': 2.19, 'PHE': 1.83, 'GLY': 2.34, 'HIS': 1.82, 'ILE': 2.36,
                 'LYS': 2.18, 'LEU': 2.36, 'MET': 2.28, 'ASN': 2.02, 'PRO': 1.99, 'GLN': 2.17, 'ARG': 2.17, 'SER': 2.21,
                 'THR': 2.09, 'VAL': 2.32, 'TRP': 2.83, 'TYR': 2.32}

res_pkb_table = {'ALA': 9.69, 'CYS': 10.28, 'ASP': 9.60, 'GLU': 9.67, 'PHE': 9.13, 'GLY': 9.60, 'HIS': 9.17,
                 'ILE': 9.60, 'LYS': 8.95, 'LEU': 9.60, 'MET': 9.21, 'ASN': 8.80, 'PRO': 10.60, 'GLN': 9.13,
                 'ARG': 9.04, 'SER': 9.15, 'THR': 9.10, 'VAL': 9.62, 'TRP': 9.39, 'TYR': 9.62}

res_pkx_table = {'ALA': 0.00, 'CYS': 8.18, 'ASP': 3.65, 'GLU': 4.25, 'PHE': 0.00, 'GLY': 0, 'HIS': 6.00,
                 'ILE': 0.00, 'LYS': 10.53, 'LEU': 0.00, 'MET': 0.00, 'ASN': 0.00, 'PRO': 0.00, 'GLN': 0.00,
                 'ARG': 12.48, 'SER': 0.00, 'THR': 0.00, 'VAL': 0.00, 'TRP': 0.00, 'TYR': 0.00}

res_pl_table = {'ALA': 6.00, 'CYS': 5.07, 'ASP': 2.77, 'GLU': 3.22, 'PHE': 5.48, 'GLY': 5.97, 'HIS': 7.59,
                'ILE': 6.02, 'LYS': 9.74, 'LEU': 5.98, 'MET': 5.74, 'ASN': 5.41, 'PRO': 6.30, 'GLN': 5.65,
                'ARG': 10.76, 'SER': 5.68, 'THR': 5.60, 'VAL': 5.96, 'TRP': 5.89, 'TYR': 5.96}

res_hydrophobic_ph2_table = {'ALA': 47, 'CYS': 52, 'ASP': -18, 'GLU': 8, 'PHE': 92, 'GLY': 0, 'HIS': -42, 'ILE': 100,
                             'LYS': -37, 'LEU': 100, 'MET': 74, 'ASN': -41, 'PRO': -46, 'GLN': -18, 'ARG': -26, 'SER': -7,
                             'THR': 13, 'VAL': 79, 'TRP': 84, 'TYR': 49}

res_hydrophobic_ph7_table = {'ALA': 41, 'CYS': 49, 'ASP': -55, 'GLU': -31, 'PHE': 100, 'GLY': 0, 'HIS': 8, 'ILE': 99,
                             'LYS': -23, 'LEU': 97, 'MET': 74, 'ASN': -28, 'PRO': -46, 'GLN': -10, 'ARG': -14, 'SER': -5,
                             'THR': 13, 'VAL': 76, 'TRP': 97, 'TYR': 63}

def generate_contact_map(pdb_file, cutoff=8.0):
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    # Extract CA (alpha carbon) atoms
    ca_atoms = []
    residue_info = []

    for model in structure:
        for chain in model:
            for residue in chain:
                try:
                    ca = residue['CA']
                    ca_atoms.append(ca.coord)
                    residue_info.append((chain.id, residue.id[1], residue.resname))
                except KeyError:
                    # Skip residues without CA atom
                    continue
    coords = np.array(ca_atoms)
    distances = squareform(pdist(coords))
    contact_map = distances <= cutoff
    np.fill_diagonal(contact_map, False)

    return contact_map, residue_info


def process_pdb_directory(input_dir, output_dir, cutoff=8.0):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith('.pdb'):
            pdb_path = os.path.join(input_dir, filename)

            try:
                contact_map, residue_info = generate_contact_map(pdb_path, cutoff)
                base_name = os.path.splitext(filename)[0]
                output_file = os.path.join(output_dir, f'{base_name}_contact_map.npz')
                np.savez(output_file,
                         contact_map=contact_map,
                         residue_info=residue_info)

                print(f'Processed {filename}: Contact map saved to {output_file}')
            except Exception as e:
                print(f'Error processing {filename}: {e}')


def load_contact_map(npz_file):
    data = np.load(npz_file, allow_pickle=True)

    residue_info = [
        (str(chain), int(residue_number), str(residue_name))
        for chain, residue_number, residue_name in data['residue_info']
    ]

    return data['contact_map'], residue_info


def calculate_pI(residue_name, pk_a_values_cooh, pk_a_values_nh3, pk_a_values_imidazole,
                 pk_a_values_sh, pk_a_values_oh):
    pka_values = []
    if residue_name not in pk_a_values_cooh or residue_name not in pk_a_values_nh3:
        raise ValueError(f"Missing required pKa values for {residue_name}")

    pka_values.extend([
        pk_a_values_cooh[residue_name],
        pk_a_values_nh3[residue_name]
    ])

    if residue_name in pk_a_values_imidazole:
        pka_values.append(pk_a_values_imidazole[residue_name])
    elif residue_name in pk_a_values_sh:
        pka_values.append(pk_a_values_sh[residue_name])
    elif residue_name in pk_a_values_oh:
        pka_values.append(pk_a_values_oh[residue_name])

    pka_values.sort()

    if len(pka_values) == 2:
        return sum(pka_values) / 2
    elif len(pka_values) == 3:
        if pka_values[-1] > 7:
            return (pka_values[-1] + pka_values[-2]) / 2
        else:
            return (pka_values[0] + pka_values[1]) / 2

    return None


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def residue_features(residue):
    res_property1 = [
        1 if residue in pro_res_aliphatic_table else 0,
        1 if residue in pro_res_aromatic_table else 0,
        1 if residue in pro_res_polar_neutral_table else 0,
        1 if residue in pro_res_acidic_charged_table else 0,
        1 if residue in pro_res_basic_charged_table else 0
    ]
    res_property2 = [
        res_weight_table.get(residue, 0), 
        res_pka_table.get(residue, 0),     
        res_pkb_table.get(residue, 0),     
        res_pkx_table.get(residue, 0),     
        res_pl_table.get(residue, 0),      
        res_hydrophobic_ph2_table.get(residue, 0),  
        res_hydrophobic_ph7_table.get(residue, 0)   
    ]

    return np.array(res_property1 + res_property2)

def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        pro_hot[i,] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i,] = residue_features(pro_seq[i])

    return np.concatenate((pro_hot, pro_property), axis=1)

def contact_map_to_graph(contact_map, residue_info, pro_seq):
    G = nx.Graph()
    sequence_features = seq_feature(pro_seq)

    for i, (_, residue_number, residue_name) in enumerate(residue_info):
        residue_name = str(residue_name).strip().upper()

        if residue_name in amino_acids_set:
            try:
                feature_vector = sequence_features[i]
                G.add_node(i,
                           residue_name=residue_name,
                           features=str(feature_vector.tolist())  # Convert the list to a string
                           )
            except Exception as e:
                print(f"Critical error processing {residue_name} at node {residue_number}: {e}")
                print(f"Full traceback:", traceback.format_exc())

    for i in range(contact_map.shape[0]):
        for j in range(i + 1, contact_map.shape[1]):
            if contact_map[i, j]:
                G.add_edge(i, j)
    return G


def plot_graph(G, title, output_path=None):
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)  
  
    unique_residues = list(set(nx.get_node_attributes(G, 'residue_name').values()))
    color_map = plt.cm.get_cmap('tab20')  # Use the 'tab20' color map
    color_dict = {res: color_map(i / len(unique_residues)) for i, res in enumerate(unique_residues)}

    node_colors = [color_dict[G.nodes[node]['residue_name']] for node in G.nodes()]

    nx.draw(G, pos,
            with_labels=False,  # Disable labels to reduce clutter
            node_size=50,
            node_color=node_colors,
            alpha=0.7,
            edge_color='gray',
            width=0.5)

    plt.title(title)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def process_contact_maps(contact_maps_dir, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(contact_maps_dir, 'graph_outputs')
    os.makedirs(output_dir, exist_ok=True)

    plot_dir = os.path.join(output_dir, 'plots')
    gml_dir = os.path.join(output_dir, 'gml')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(gml_dir, exist_ok=True)

    for filename in os.listdir(contact_maps_dir):
        if filename.endswith('_contact_map.npz'):
            npz_path = os.path.join(contact_maps_dir, filename)
            try:
                print(f"Processing file: {filename}")

                contact_map, residue_info = load_contact_map(npz_path)
                pro_seq = [residue_name for _, _, residue_name in residue_info]

                G = contact_map_to_graph(contact_map, residue_info, pro_seq)

                base_name = os.path.splitext(filename)[0]
                plot_path = os.path.join(plot_dir, f'{base_name}_graph.png')
                gml_path = os.path.join(gml_dir, f'{base_name}_graph.gml')

                plot_graph(G, base_name, plot_path)

                nx.write_gml(G, gml_path)

                print(f'Processed {filename}:')
                print(f'  Nodes: {G.number_of_nodes()}')
                print(f'  Edges: {G.number_of_edges()}')
                print(f'  Plot saved to: {plot_path}')
                print(f'  GML saved to: {gml_path}\n')

                for node in G.nodes():
                    features = G.nodes[node].get('features', [])
                    print(f"Node {node} features: {features[:5]}...")  # Preview first 5 features

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                print(f"Full traceback:", traceback.format_exc())


if __name__ == '__main__':
    input_directory = 'path_to_pdb_files'
    contact_maps = 'path_to_the_contact_maps'
    output_dir = 'path_to_the_dir_for_protein_graphs'

    process_pdb_directory(input_directory, contact_maps)
    process_contact_maps(contact_maps, output_dir)
