import hycom_nodes
from ssted import tnet
from ssted import network_writer
from ssted import network_measures
from ssted import draw_measures
import gcoss_nodes

def main():
    hycom_list = hycom_nodes.get_nodelist()
    gcoos_df = gcoss_nodes.get_gcoos_dataframe()
    nodes_df_list = [ gcoos_df[['Lon','Lat']].copy() ] * 7 
    edges_df_list = []
    for time in range(7):
        print(f"time: [{time}]")
        roi_snapshot = hycom_nodes.get_nodes_roi_at_time(hycom_list, time)
        edges_df = hycom_nodes.get_edgelist(roi_snapshot)
        edges_df_list.append(edges_df)
    tn = tnet.from_list_of_dataframes(edges_df_list, nodes_df_list)
    print(f"nodes: {len(tn.nodes)}, edges: {len(tn.edges)}")
    print(f"tnet: {str(tn)}")
    #tn = tnet.get_tnet()
    #network_writer.save_json(tn,name='tnet',start=0,end=len(tn))
    analyze(tn)
    analyze_temporal_coverage(edges_df_list)


def analyze(tn): 
    degrees = network_measures.temporal_degree_centrality(tn)
    draw_measures.draw_temporal_degree_centrality(degrees)
    print('degrees')
    degree_totals = network_measures.temporal_degree_centrality_overall(degrees)
    draw_measures.draw_temporal_degree_centrality_overall(degree_totals)
    print('degree totals')
    overlaps_network = network_measures.topological_overlap_overall(tn)
    draw_measures.draw_topological_overlap_overall(overlaps_network)
    print('overlaps-network')
    overlaps_nodes = network_measures.topological_overlap(tn)
    draw_measures.draw_topological_overlap(overlaps_nodes)
    print('overlaps-nodes')
    overlap_averages = network_measures.topological_overlap_average(overlaps_nodes)
    draw_measures.draw_topological_overlap_average(overlap_averages)
    print('overlap averages')
    tcc = network_measures.temporal_correlation_coefficient(overlap_averages)
    draw_measures.draw_temporal_correlation_coefficient(tcc)
    print('tcc', tcc)


def analyze_temporal_coverage(edges_df_list):
    scores = []
    for edges_df in edges_df_list:
        #print('edge_df: ',edges_df)
        coverage_score = edges_df[['Dist']].sum()
        scores.append(coverage_score)
    total_score = sum(scores)
    avg_score = total_score / len(scores)
    #print('scores: ', scores)
    #print('total score: ', total_score)
    #print('avg_score: ', avg_score)
    return {'scores': scores, 'total_score': total_score, 'average_score': avg_score}


def analyze_coverage(edges_df):
    coverage_score = edges_df[['Dist']].sum()
    return coverage_score




if __name__ == "__main__":
    main()