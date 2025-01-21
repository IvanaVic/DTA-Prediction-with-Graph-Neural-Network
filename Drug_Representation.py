from rdkit import Chem
import matplotlib
import os
import pandas as pd
import networkx as nx
import numpy as np
import json
import matplotlib.pyplot as plt
import re
import hashlib

matplotlib.use('Agg')

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print(f"Error: Invalid SMILES string: {smile}")
        return None
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))  # Normalize features

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    g = nx.Graph()
    g.add_nodes_from(range(c_size)) 
    g.add_edges_from(edges) 

    for i, feature in enumerate(features):
        g.nodes[i]['feature'] = ','.join(map(str, feature))
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])

    return c_size, features, edge_index, g


def sanitize_filename_KIBA(smile):
    smile_hash = hashlib.md5(smile.encode()).hexdigest()
    return smile_hash

def sanitize_filename_DAVIS(smile):
    return re.sub(r'[\\/*?:"<>|]', '_', name)


def process_ligands_from_dataset(file_path, output_dir):
    data = pd.read_csv(file_path)
    gml_dir = os.path.join(output_dir, "gml_files2_KIBA")
    plot_dir = os.path.join(output_dir, "plots2_KIBA")
    os.makedirs(gml_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    ligand_target_mapping = {}

    graphs = {}

    for _, row in data.iterrows():
        print(f"Processing row: {row}")
        smile = row['ligand']
        target = row['protein']
        label = row['label']

        ligand_name = sanitize_filename_DAVIS(smile)  
        if ligand_name not in graphs:
            result = smile_to_graph(smile)
            if result:
                c_size, features, edge_index, g = result
                graphs[ligand_name] = {
                    "c_size": int(c_size),  
                    "features": np.array(features).tolist(), 
                    "edge_index": [[int(i), int(j)] for i, j in edge_index],
                }

                gml_file = os.path.join(gml_dir, f"{ligand_name}.gml")
                nx.write_gml(g, gml_file)

                plt.figure(figsize=(8, 8))
                pos = nx.spring_layout(g)
                nx.draw(g, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=8, font_weight="bold")
                plt.title(f"Graph for {ligand_name}")
                plot_file = os.path.join(plot_dir, f"{ligand_name}.png")
                plt.savefig(plot_file)
                plt.close()

                print(f"Saved graph for {ligand_name}: GML -> {gml_file}, Plot -> {plot_file}")

        if ligand_name not in ligand_target_mapping:
            ligand_target_mapping[ligand_name] = []

        ligand_target_mapping[ligand_name].append({
            "protein": target,
            "label": label
        })

    output_path = os.path.join(output_dir, "ligand_graphs2.json")
    with open(output_path, 'w') as f:
        json.dump({
            "graphs": graphs,
            "ligand_target_mapping": ligand_target_mapping
        }, f, indent=4)

    print(f"Graph data saved to {output_path}")

file_path = "path_to_the_dataset"
output_dir = "path_to_the_output_dir"

process_ligands_from_dataset(file_path, output_dir)
