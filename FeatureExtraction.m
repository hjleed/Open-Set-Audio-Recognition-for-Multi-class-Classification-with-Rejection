function features=FeatureExtraction(audioIn,fs)
 windowLength  = 2048;
 overlapLength = 512;
 win = hamming(windowLength,"periodic");
 %%  Silence Removal
audioIn=Silence_Removal(audioIn,windowLength,overlapLength);
 
 afe = audioFeatureExtractor( ...
    'Window',       win, ...
    'OverlapLength',overlapLength, ...
    'SampleRate',   fs, ...
    ...
    'linearSpectrum',      false, ...
    'melSpectrum',         false, ...
    'barkSpectrum',        false, ...
    'erbSpectrum',         false, ...
    ...
    'mfcc',                true, ...
    'mfccDelta',           true, ...
    'mfccDeltaDelta',      true, ...
    'gtcc',                true, ...
    'gtccDelta',           true, ...
    'gtccDeltaDelta',      true, ...
    ...
    'spectralCentroid',    true, ...
    'spectralCrest',       true, ...
    'spectralDecrease',    false, ...
    'spectralEntropy',     false, ...
    'spectralFlatness',    false, ...
    'spectralFlux',        true, ...
    'spectralKurtosis',    false, ...
    'spectralRolloffPoint',true, ...
    'spectralSkewness',    true, ...
    'spectralSlope',       true, ...
    'spectralSpread',      true, ...
    ...
    'pitch',               false, ...
    'harmonicRatio',       false);

%%
 setExtractorParams(afe,"spectralRolloffPoint","Threshold",0.95)
 setExtractorParams(afe,"mfcc","NumCoeffs",13)
 features = extract(afe,audioIn);
 %idx = info(afe)
end

function audioIn=Silence_Removal(audioIn,windowLength,overlapLength)
[segments,~] = buffer(audioIn,windowLength,overlapLength,'nodelay');
audioIn = audioIn/max(abs(audioIn));

end
