export type SinkClassification = "detrimental" | "neutral" | "beneficial";

export type AnalysisFormValues = {
  modelName: string;
  text: string;
};

export type AnalyzeRequest = {
  model_name: string;
  text: string;
};

export type AnalyzeResponse = {
  token_list: string[];
  corrected_att_scores: number[][][][][];
  classifications: SinkClassification[];
  att_received_scores: number[];
  value_norms: number[];
};

export type TokenAnalysisRow = {
  index: number;
  token: string;
  classification: SinkClassification;
  attentionReceived: number;
  valueNorm: number;
};
