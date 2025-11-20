

edfFilePath = 'C:\Users\melis\Documents\Signals Project\data\Training'; 
xmlFilePath = edfFilePath;
[hdr, record] = read_edf(edfFilePath); 

%%

r = 1;  % 1 = R1, 2 = R2, ..., 10 = R10

% Choose which record to plot (e.g. R1 -> index 1)
r = 10;
recName = sprintf('R%d', r);

% Load stages from XML for that record
xmlFilename = fullfile(xmlFilePath, [recName '.xml']);
[events, stages, epochLength, annotation] = readXML(xmlFilename);

% Call your plotting function
plot_record_with_stages(hdr{r}, record{r}, stages, epochLength);


%%

clear all
clc


edfFilePath = 'C:\Users\melis\Documents\Signals Project\data\Training'; 
xmlFilePath = edfFilePath; 
[eeg_data, labels] = load_training_data(edfFilePath, xmlFilePath); 