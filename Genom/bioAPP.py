import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
import io
import pandas as pd

import pickle
import py3Dmol
import sklearn
from stmol import showmol

matplotlib.use("Agg")
from Bio.Seq import Seq
from Bio import SeqIO
from collections import Counter
# import neatbio.sequtils as utils
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import numpy as np
from PIL import Image
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB import *
import os


def delta(x, y):
    return 0 if x == y else 1


def M(seq1, seq2, i, j, k):
    return sum(delta(x, y) for x, y in zip(seq1[i:i + k], seq2[j:j + k]))


def makeMatrix(seq1, seq2, k):
    n = len(seq1)
    m = len(seq2)
    return [[M(seq1, seq2, i, j, k) for j in range(m - k + 1)] for i in range(n - k + 1)]


def plotMatrix(M, t, seq1, seq2, nonblank=chr(0x25A0), blank=' '):
    print(' |' + seq2)
    print('-' * (2 + len(seq2)))
    for label, row in zip(seq1, M):
        line = ''.join(nonblank if s < t else blank for s in row)
        print(label + '|' + line)


def dotplot(seq1, seq2, k=1, t=1):
    M = makeMatrix(seq1, seq2, k)
    plotMatrix(M, t, seq1, seq2)  # experiment with character choice


# Convert to Fxn
def dotplotx(seq1, seq2):
    plt.imshow(np.array(makeMatrix(seq1, seq2, 1)))
    # on x-axis list all sequences of seq 2
    xt = plt.xticks(np.arange(len(list(seq2))), list(seq2))
    # on y-axis list all sequences of seq 1
    yt = plt.yticks(np.arange(len(list(seq1))), list(seq1))
    plt.show()


def FullName(seq):
    dic = {'A': 'Alanine', 'R': 'Arginine', 'N': 'Asparagine', 'D': 'Aspartic acid', 'C': 'Cysteine',
           'E': 'Glutamic acid', 'Q': 'Glutamine', 'G': 'Glycine', 'H': 'Histidine', 'I': 'Isoleucine', 'L': 'Leucine',
           'K': 'Lysine', 'M': 'Methionine', 'F': 'Phenylalanine', 'P': 'Proline', 'S': 'Serine', 'T': 'Threonine',
           'W': 'Tryptophan', 'Y': 'Tyrosine', 'V': 'Valine', 'U': 'Selenocysteine', 'O': 'Pyrrolysine'}
    aa_name = str(seq).replace("*", "")
    sperator = ","
    aa2_name = sperator.join(aa_name)
    list = aa2_name.split(',')
    i = 0
    result = ''
    for i in list:
        for key in dic.keys():
            if i == key:
                result += dic[key] + ', '
    return result


def makeblock(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    mblock = Chem.MolToMolBlock(mol)
    return mblock


def render_mol(xyz):
    xyzview = py3Dmol.view(width=650, height=250)
    xyzview.addModel(xyz, 'mol')
    xyzview.setStyle({'stick': {}})
    xyzview.setBackgroundColor('white')
    xyzview.zoomTo()
    showmol(xyzview, height=250, width=500)


def AromaticProportion(m):
    aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
    aa_count = []
    for i in aromatic_atoms:
        if i == True:
            aa_count.append(1)
    AromaticAtom = sum(aa_count)
    HeavyAtom = Descriptors.HeavyAtomCount(m)
    AR = AromaticAtom / HeavyAtom
    return AR


def generate(smiles, verbose=False):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)
    baseData = np.arange(1, 1)
    i = 0
    for mol in moldata:
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        desc_AromaticProportion = AromaticProportion(mol)
        row = np.array([desc_MolLogP,
                        desc_MolWt,
                        desc_NumRotatableBonds,
                        desc_AromaticProportion])
        if i == 0:
            baseData = row
        else:
            baseData = np.vstack([baseData, row])
        i = i + 1
    columnNames = ["MolLogP", "MolWt", "NumRotatableBonds", "AromaticProportion"]
    descriptors = pd.DataFrame(data=baseData, columns=columnNames)
    return descriptors


def main():
    """A Simple Streamlit App """
    st.title("POMICS")

    activity = ['Intro', 'DNA Sequence','COVID-19 Protein Sequence Analysis',  'DotPlot',
                'Molecular Prediction and visualization', "About"]
    choice = st.sidebar.selectbox("Select Activity", activity)
    if choice == 'Intro':
        st.header("Intro")

        st.image('proteinsynthesis01.png')
        st.subheader("What is POMICS ?!")
        st.write("POMICS is a protein sequence analysis tool that tak DNA file in the FASTA format .")
        st.write(" This file will be transcripted and translated into "
                 "its corresponding protein sequence , analyses it and represent"
                 " data about this file using plotting and visualization ")
        st.write("It also represent molecular solubility prediction and visualization using Machine Learning Model")
    elif choice == 'COVID-19 Protein Sequence Analysis':
        st.image('proteinsynthesis02.jpg')
        ncov_dna_record = SeqIO.read("sequence.fasta", "fasta")
        ncov_dna = ncov_dna_record.seq
        st.subheader("COVID SEQUENCE")
        st.write(ncov_dna[0:800], )
        st.subheader("Length of COVID seq:")
        st.write(len(ncov_dna))
        # Nucleotide Frequencies
        st.subheader("Nucleotide Frequency")
        dna_freq1 = Counter(ncov_dna)
        for key, value in dna_freq1.items():
            st.write("{}: {}".format(key, value))
        st.subheader("Nucleotide Frequency Plot")
        barlist = plt.bar(dna_freq1.keys(), dna_freq1.values())
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.subheader("Transcribe (DNA to mRNA)")
        ncov_mRNA = ncov_dna.transcribe()
        st.write(ncov_mRNA[0:900])
        scodn = "AUG"
        index = ncov_mRNA.find(scodn)
        st.subheader("Start coden index:")
        st.write(index)
        st.subheader('Translate to Protein/Amino Acids (mRNA to AA)')
        ncov_protein = ncov_mRNA.translate()
        st.write(ncov_protein[0:990])
        st.subheader("Length of Protin without stop codon(*)::Amino Acids :")
        name = str(ncov_protein).replace("*", "")
        st.write(len(name))
        ncov_amino_acids = ncov_protein.split('*')
        ncov_aa = [str(i) for i in ncov_amino_acids]
        st.subheader('Place our Polypeptids into a DataFrame')
        df = pd.DataFrame({'amino_acids/polypeptids chains': ncov_aa})
        st.write(df)
        df['count'] = df['amino_acids/polypeptids chains'].apply(len)
        st.write(df)
        st.write('The Number of Polypeptidies Chains')
        st.write(len(ncov_amino_acids))
        st.subheader('The largest 15 amino acid (Polypeptids) sequence')
        st.write(df['count'].nlargest(15))
        st.write(df.nlargest(15, 'count'))
        st.subheader("Full Amino Acids Names")
        res = FullName(ncov_protein)
        st.write(res[0:990])
        name = str(ncov_protein).replace("*", "")
        c = Counter(str(name))
        st.subheader("AA Frequancy")
        for key, value in c.items():
            st.write("{}: {}".format(key, value))
        plt.bar(c.keys(), c.values())
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        c1 = c.most_common(1)
        st.write("//Small Help :: Each amino acid with it's apreviation")
        st.write(
            "A = alanine, R=arginine, N = asparagine, D= aspartic acid, C =cysteine, E = glutamic acid, Q = glutamine, G = glycine, H = histidine, I = isoleucine, L = leucine, K = lysine, M = methionine, F = phenylalanine, P = proline, S = serine, T = threonine, W = tryptophan, Y = tyrosine,V = valine, U = selenocysteine, O = pyrrolysine")
        st.subheader("Top Most Common Amino")
        st.write(c1[0][0:2], " Leucine")
        # Chains in the Protein Structure
        # parser = MMCIFParser()
        # structure = parser.get_structure('6lu7','6lu7.cif')
        # xyzview = py3Dmol.view(structure)
        # xyzview.setStyle({'stick':{}})
        # xyzview.setBackgroundColor('white')
        # xyzview.zoomTo()
        # showmol(xyzview, height=500, width=800)

    elif choice == "DNA Sequence":
        st.subheader("DNA Sequence Analysis")
        seq_file = st.file_uploader("Upload FASTA File", type=["fasta", "fa"])
        if seq_file is not None:
            byte_str = seq_file.read()
            text_obj = byte_str.decode('UTF-8')
            dna_record = SeqIO.read(io.StringIO(text_obj), "fasta")
            # st.write(dna_record)
            dna_seq = dna_record.seq
            details = st.radio("Details", ("Description", "Sequence"))
            if details == "Description":
                st.write(dna_record.description)
            elif details == "Sequence":
                st.write(dna_record.seq)

            # Nucleotide Frequencies
            st.subheader("Nucleotide Frequency")
            dna_freq = Counter(dna_seq)
            for key, value in dna_freq.items():
                st.write("{}: {}".format(key, value))
            adenine_color = st.color_picker("Adenine Color")
            thymine_color = st.color_picker("thymine Color")
            guanine_color = st.color_picker("Guanine Color")
            cytosil_color = st.color_picker("cytosil Color")

            if st.button("Plot Freq"):
                barlist = plt.bar(dna_freq.keys(), dna_freq.values())
                barlist[2].set_color(adenine_color)
                barlist[3].set_color(thymine_color)
                barlist[1].set_color(guanine_color)
                barlist[0].set_color(cytosil_color)
                st.pyplot()
                st.set_option('deprecation.showPyplotGlobalUse', False)

            # Protein Synthesis
            st.subheader("Protein Synthesis")
            p1 = dna_seq.translate()
            p2 = str(p1).replace("*", "")
            aa_freq = Counter(str(p2))

            if st.checkbox("Transcription"):
                st.write(dna_seq.transcribe())

            elif st.checkbox("Translation"):
                st.write(dna_seq.translate())

            elif st.checkbox("Place our Amino Acids into a DataFrame"):
                p2 = p1.split('*')
                ncov_aa = [str(i) for i in p2]
                df1 = pd.DataFrame({'amino_acids': ncov_aa})
                df1['count'] = df1['amino_acids'].apply(len)
                st.write(df1)

            elif st.checkbox("Complement"):
                st.write(dna_seq.complement())

            elif st.checkbox("Full Amino Acid Name"):
                st.write(FullName(p1))


            elif st.checkbox("AA Frequency"):
                for key, value in aa_freq.items():
                    st.write("{}: {}".format(key, value))

            elif st.checkbox("Plot AA Frequency"):
                aa_color = st.color_picker("Pick An Amino Acid Color")
                plt.bar(aa_freq.keys(), aa_freq.values(), color=aa_color)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

            elif st.checkbox(" Amino Acid Percent"):
                st.write("//Small Help :: Each amino acid with it's apreviation")
                st.write(
                    "A = alanine, R=arginine, N = asparagine, D= aspartic acid, C =cysteine, E = glutamic acid, Q = glutamine, G = glycine, H = histidine, I = isoleucine, L = leucine, K = lysine, M = methionine, F = phenylalanine, P = proline, S = serine, T = threonine, W = tryptophan, Y = tyrosine,V = valine, U = selenocysteine, O = pyrrolysine")
                p1_analysed = ProteinAnalysis(str(p1))
                ac_count = st.text_input("Enter The Amino Acid You Want To Know It's Percentage Here",
                                         "Write AMINO Abbreviation In Capital Letter")
                if ac_count in str(p1):
                    st.write("Percent of {} Amino is :: {}".format((ac_count),
                                                                   p1_analysed.get_amino_acids_percent()[ac_count]))
                elif ac_count == "Write AMINO Abbreviation In Capital Letter":
                    st.write('')
                else:
                    st.write('this "{}" Amino Acide is not found in the sequence'.format(ac_count))

            elif st.checkbox("Top Most Common Amino"):
                c = aa_freq.most_common(1)
                st.write(c[0][0:2])

    elif choice == "DotPlot":
        st.subheader("Generate Dot Plot For Two Sequences")
        seq_file1 = st.file_uploader("Upload 1st FASTA File", type=["fasta", "fa"])
        seq_file2 = st.file_uploader("Upload 2nd FASTA File", type=["fasta", "fa"])

        if seq_file1 and seq_file2 is not None:
            byte_str1 = seq_file1.read()
            text_obj1 = byte_str1.decode('UTF-8')
            dna_record1 = SeqIO.read(io.StringIO(text_obj1), "fasta")
            byte_str2 = seq_file2.read()
            text_obj2 = byte_str2.decode('UTF-8')
            dna_record2 = SeqIO.read(io.StringIO(text_obj2), "fasta")
            # st.write(dna_record)
            dna_seq1 = dna_record1.seq
            dna_seq2 = dna_record2.seq

            details = st.radio("Details", ("Description", "Sequence"))
            if details == "Description":
                st.write(dna_record1.description)
                st.write("lengh of seq1: ", len(dna_seq1))
                st.write("=====================")
                st.write(dna_record2.description)
                st.write("lengh of seq2: ", len(dna_seq2))
            elif details == "Sequence":
                st.write(dna_record1.seq)
                st.write("=====================")
                st.write(dna_record2.seq)
            st.subheader("Alignment Score")
            seq1_n_seq2 = pairwise2.align.globalxx(dna_seq1, dna_seq2, one_alignment_only=True, score_only=True)
            st.write(seq1_n_seq2)
            cus_limit = st.number_input("Select Max number of Nucleotide for dotplot", 10, 200, 30)
            if st.button("Dot Plot"):
                st.write("Comparing the first {} Nucleotide of the Two Sequences".format(cus_limit))
                dotplotx(dna_seq1[0:cus_limit], dna_seq2[0:cus_limit])
                st.pyplot()
                st.set_option('deprecation.showPyplotGlobalUse', False)
    elif choice == 'Molecular Prediction and visualization':
        image = Image.open('solubility-logo.jpg')

        st.image(image, use_column_width=True)
        st.write("""
                This  predicts the (LogS) values of molecules!
                """)
        compound_smiles = st.text_input('SMILES please', 'CC')
        blk = makeblock(compound_smiles)
        render_mol(blk)
        st.write("""
        # Molecular Prediction 
        """)

        st.subheader('User Input Features')
        SMILES_input = "NCCCC\nCCC\nCN"

        SMILES = st.text_area("SMILES input", SMILES_input)
        SMILES = "C\n" + SMILES
        SMILES = SMILES.split('\n')
        st.header('Computed molecular descriptors')
        X = generate(SMILES)
        X[1:]
        load_model = pickle.load(open('solubility_model.pkl', 'rb'))
        # Apply model to make predictions
        prediction = load_model.predict(X)
        st.header('Predicted LogS values')
        prediction[1:]
    elif choice == "About":
        st.subheader("About")
        st.write("Contact us ")
        st.write(" Tasneemmohamed1811@gmail.com")
        st.write(" Nadin.Ahmed.2023@gmail.com")
        st.write(" ameraalsa3id@gamail.com")
        st.write(" iyomnaothman@gamil.com")
        st.write(" fatmaragababdlaal@gamil.com")

if __name__ == '__main__':
    main()
