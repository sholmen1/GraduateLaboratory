%%%% SEANS UPDATED testFitFunctionANN_1D_studentVersion.m FILE   %%%%
    close all;
    plotLabels = 'a':'z';

    % initialize the figure
    figure('Color','w','Units','normalized','Position',[0,0.5,1,0.5]);
    tiledlayout(5,2);

    % number of grid points
    N = 32;

    % number of coefficients, including x.^0
    n = 6;
    
    % construct simulation grid
    %
    % maximum extent of grid
    xMax = 2;

    % pixel size of grid
    dx = xMax/N;

    % simulation grid
    x = -xMax/2:dx:xMax/2-dx;

    % maximum extent of grid
    fxMax = 1/dx;

    % pixel size of grid
    dfx = fxMax/N;

    % simulation grid
    fx = -fxMax/2:dfx:fxMax/2;
    
    % ANN parameters
    trainingIterations = 1e4;
    learnRateW = 0.7;
    learnRateB = 0.7;
    
% ===================================================================== %
% polynomial fitting                                                    %
% ===================================================================== %

    %%% coefficient vector (random)
    coeffs = 50*(rand(1,n)-0.5)';

    % X - matrix: does NOT include x.^0 term. The bias vector takes care of
    % that
    inputs1 = (repmat(x(:),1,n-1).^(1:n-1))';

    % f = X * a: DOES include x.^0 term
    targets = (repmat(x(:),1,n).^(0:n-1) * coeffs)';

    % running the ANN
    [weights,bias,pltHandle] = fitFunctionANN_1D_studentVersion(targets,inputs1,learnRateW,learnRateB,trainingIterations);

    % delete the error figure
    close(ancestor(pltHandle,'figure'));
    
    % reconstruct the function with the 'learned' coefficients
    f2 = weights*inputs1+bias;

    % calculate R^2 value
    Rsqrd = 1-sum( (targets-f2).^2 )/sum( (targets-mean(targets)).^2 );

% plot ---------------------------------------------------------------- %
    ax = nexttile(1);
    line(x,targets,'color','r','lineWidth',2);
    line(x,f2,'color','b','marker','d');
    box on; axis tight;
    xl = xlabel('x');
    yl = ylabel('f(x)','Rotation',0);
    tl = title('Function and Fit');
    
    txl = makeTxtLabel(ax,plotLabels(1));
    txlRsqrd = makeTxtLabel(ax,['  R^2: ', num2str(Rsqrd)],1);
    set([gca,xl,yl,tl,txl,txlRsqrd],'fontSize',18,'fontWeight','bold');
    
%%bar plot 
    ax = nexttile(2);
    tl.String = 'Exactly-Determined';
    a = coeffs;
    b = [bias,weights]';
    c = a-b;
    bar([a,b,c]);

    xl = xlabel('coeff #');
    yl = ylabel('coeff');
    legend({'True','Fit','Error'})
    txl = makeTxtLabel(ax,plotLabels(2));
    set([gca,xl,yl,txl],'fontSize',18,'fontWeight','bold');

    ratio_EX = n/length(inputs1);  % number true coefficients/number of columns in 
fprintf('Rows to columns ratio: %.1f\n', ratio_EX);
fprintf('R$^{2}$ value: %.6f\n', Rsqrd);

% if ratio_EX > 1
%     fprintf('System is Overdetermined (more_equations than unknowns) \n');
% elseif ratio_EX < 1
%     fprintf('System is Underdetermined (fewer equations than unknowns) \n')
% else
%     fprintf('System is EXACTLY-DETERMINED (equal equations and unknowns)\n');
% end

% Add this section to your existing code after the first two plots
% This creates plots (c) and (d) for the over-determined system

% ===================================================================== %
% Over-determined polynomial fitting (MORE equations than coefficients) %
% ===================================================================== %

% For over-determined system: Use 6th order polynomial to generate data
% but fit with only 8th order polynomial (fewer coefficients than data complexity)

% Generate a 6th order polynomial
n_order = 6; 
true_coeffs = 50 *(rand(1, n_order)-0.5)'; 

%
x_True = repmat(x(:), 1, n_order) .^ (0:n_order-1); %does include x^0 term
f_True = x_True * true_coeffs; % f = X * a 

%Fitting function using 8th order polynomial ie underfitted
n_fit = 8; 

%ANN targets/inputs(underfit) overdetermined less true coefficents than fit (exludes x^0 via bias term)
inputs_ANN_UF = (repmat(x(:), 1, n_fit-1).^(1: n_fit-1))'; 
targets_ODC = f_True'; %Target is the complex 6th order polynomial 

%Running the ANN for the weight, matrix, bias vector, and plothandle I
[weights_ODC, bias_ODC, pltHandle_ODC] = fitFunctionANN_1D_studentVersion(targets_ODC, inputs_ANN_UF, learnRateW, learnRateB, trainingIterations);

%Close the error figure, oh this is the figure calculating the errors before
%final plot I think
close(ancestor(pltHandle_ODC, 'figure'))

%Reconstruct Fitted Function via ANN parameters 
f_Fit = weights_ODC*inputs_ANN_UF + bias_ODC;

% calculate R^2 value over determined case
Rsqrd_ODC = 1-sum((targets_ODC-f_Fit).^2 )/sum( (targets_ODC-mean(targets_ODC)).^2 );

%Plot c - 8th Order Function and Fit for ODC
ax = nexttile(3);
line(x, targets_ODC, 'color', 'r', 'lineWidth',2);
line(x, f_Fit, 'color', 'b', 'marker', 'd', 'MarkerSize', 4);
box on; axis tight;
xl = xlabel('x');
yl =ylabel('f(x)', 'Rotation',0);
tl = title('Over-Determined System');

txl = makeTxtLabel(ax, plotLabels(3));
txlRsqrd = makeTxtLabel(ax, ['R^2:', num2str(Rsqrd_ODC, '%.3f')],1);
set([gca,xl,yl,tl,txl,txlRsqrd], 'fontSize',18,'fontWeight','bold');

%Plot (d) - Coefficient comparison for ODC
ax=nexttile(4);

% True Coefficients 
% Instead I will try padding first the a to see if i can get the full bar graph
a_True_Subset = [true_coeffs;zeros(2,1)];
%Fitted Coefficents
b_Fit = [bias_ODC, weights_ODC]';
%Error between true and fitted coefficents 
c_Error = a_True_Subset - b_Fit;

%Create the bar plot 
bar_handles = bar([a_True_Subset, b_Fit, c_Error]);
set(bar_handles(1), 'FaceColor', [0, 0.4470, 0.7410]); %Blue
set(bar_handles(2), 'FaceColor', [0.8500 0.3250 0.0980]); % Red  
set(bar_handles(3), 'FaceColor', [0.9290 0.6940 0.1250]); % Yellow

xl = xlabel('Coefficient #');  %xlabel
yl = ylabel('Coefficient Value'); %ylabel
tl = title('Over-Determined Coefficients'); %title
legend({'True (subset)', 'Fitted', 'Error'}, 'Location', 'best'); %legend

txl = makeTxtLabel(ax,plotLabels(4));   
%ie PlotLabels is literally the alphabet and plotLabels(4) is the letter d 
% setting the plot by getting current axes, and decorating it with x+y+title labels, and letter d
set([gca,xl,yl,tl,txl],'fontSize',18,'fontWeight','bold');

%Display Informaton

%Text about System
rows_Cols_Ratio = n_order/length(inputs_ANN_UF);

%Analysis
fprintf('Rows to columns ratio: %.1f\n', rows_Cols_Ratio);
fprintf('R$^{2}$ value: %.6f\n', Rsqrd_ODC);

% if rows_Cols_Ratio > 1
%     fprintf('System is Overdetermined (more_equations than unknowns) \n');
% elseif rows_Cols_Ratio < 1
%     fprintf('System is Underdetermined (fewer equations than unknowns) \n')
% else
%     fprintf('System is EXACTLY-DETERMINED (equal equations and unknowns)\n');
% end

% ===================================================================== %
% Under-determined polynomial fitting (Less equations than coefficients) %
% ===================================================================== %
% I have some questions about this case, I am unsure if this will run the
% way I think it should 
n_UDC = 6; % 6 order includes x^0
coeffs_UDC = 50*(rand(1,n_UDC)-0.5)'; %same setup as exaxtly determined

%targets includes the x^0 term, ie the original function 6 order 
x_True_Ufit =repmat(x(:), 1, n_UDC).^(0:n_UDC-1);
targets_UDC = (x_True_Ufit*coeffs_UDC)'; 

%Create my Equation Matrix X with a 5th order polynomial fit into the ANN
%that does not include the x^0 term
n_Fit2 = 5;
x_UDC = (repmat(x(:), 1, n_Fit2-1).^(1:n_Fit2-1))'; %inputs to ANN

% Run ANN (same parameters)
[weights_UDC,bias_UDC,pltHandle_UDC] = fitFunctionANN_1D_studentVersion(targets_UDC,x_UDC,learnRateW,learnRateB,trainingIterations);
close(ancestor(pltHandle_UDC,'figure'));

% Reconstruct function
f2_UDC = weights_UDC*x_UDC+bias_UDC;

% Calculate R^2
Rsqrd_UDC = 1-sum( (targets_UDC-f2_UDC).^2 )/sum( (targets_UDC-mean(targets_UDC)).^2 );

% Plot (e) -
ax = nexttile(5);
line(x, targets_UDC, 'color', 'r', 'Marker', '_', 'lineWidth', 2);
line(x, f2_UDC, 'color', 'b', 'lineWidth', 1, 'Marker', 'd', 'MarkerSize', 8);
box on; axis tight;
xl=xlabel('x');
yl=ylabel('f(x)', 'Rotation',0);
tl=title('UnderDetermined System');

txl = makeTxtLabel(ax, plotLabels(5));
txlRsqrd = makeTxtLabel(ax, [' R^2:', num2str(Rsqrd_UDC)],1);
set([gca,xl,yl,tl,txl,txlRsqrd],'fontSize',18,'fontWeight','bold');

% Plot (f) - We will see if this is correct or not 
ax = nexttile(6);
a_UDC = coeffs_UDC;

%size(bias_UDC)            Just sanity check 
%size(weights_UDC)

b_UDC = [bias_UDC, weights_UDC, zeros(1,1) ]';
c_UDC = a_UDC - b_UDC;
bar([a_UDC, b_UDC, c_UDC]);

xl = xlabel('coeff #');
yl = ylabel('coeff');
tl = title('UnderDetermined Coefficients');
legend({'True','Fit','Error'});

txl = makeTxtLabel(ax,plotLabels(6));
set([gca,xl,yl,txl, tl],'fontSize',18,'fontWeight','bold');

% Add ratio text
ratio_UDC = n_UDC/length(x_UDC);

%Analysis
fprintf('Rows to columns ratio: %.1f\n', ratio_UDC);
fprintf('R$^{2}$ value: %.6f\n', Rsqrd_UDC);

% if ratio_UDC > 1
%     fprintf('System is Overdetermined (more_equations than unknowns) \n');
% elseif ratio_UDC < 1
%     fprintf('System is Underdetermined (fewer equations than unknowns) \n')
% else
%     fprintf('System is EXACTLY-DETERMINED (equal equations and unknowns)\n');
% end


% ===================================================================== %
% 12th order polynomial fitting to non-polynomial function             %
% ===================================================================== %

% Define the non-polynomial function as given in the problem
f_nonpoly = (cos(2*pi*6*x/xMax).*exp(-5*abs(x)/xMax) + sin(2*pi*3*x/xMax).*exp(-2*abs(x)/xMax));

% Use 12th order polynomial for fitting
n_poly = 12; % 12th order polynomial  (including x^0)

% Set up inputs and targets for ANN (excluding x^0 term in inputs, bias handles that)
inputs_NP = (repmat(x(:),1,n_poly-1).^(1:n_poly-1))'; % x^1 to x^11
targets_NP = f_nonpoly'; % Target is the non-polynomial function

% Run the ANN for polynomial fitting
[weights_NP,bias_NP,pltHandle_NP] = fitFunctionANN_1D_studentVersion(targets_NP,inputs_NP,learnRateW,learnRateB,trainingIterations);

% Close the error figure
close(ancestor(pltHandle_NP,'figure'));

% Reconstruct the fitted polynomial function
f_NP_fit = weights_NP * inputs_NP + bias_NP;

% Calculate R^2 value
Rsqrd_NP = 1-sum((targets_NP'-f_NP_fit).^2)/sum((targets_NP-mean(targets_NP)).^2);

% Calculate the error (absolute difference)
error_NP = abs(f_nonpoly - f_NP_fit);

% Plot (g) - Original function vs 12th order polynomial fit
nexttile(7)
line(x,f_nonpoly,'color','r','lineWidth',2);
line(x,f_NP_fit,'color','b','marker','d');
box on; axis tight;
xl = xlabel('x');
yl = ylabel('f(x)','Rotation',0);
tl = title('12th Order Polynomial Fit');
txlRsqrd = makeTxtLabel(gca,['  R^2: ', num2str(Rsqrd_NP)],1); 

txl = makeTxtLabel(gca,plotLabels(7));
set([gca,xl,yl,tl,txl,txlRsqrd],'fontSize',18,'fontWeight','bold');

% Plot (h) - Error between original and fitted function
nexttile(8)
line(x,error_NP,'color','k','lineWidth',2);
box on; axis tight;
xl = xlabel('x');
yl = ylabel('|Error|','Rotation',90);
tl = title('Fitting Error');
txl = makeTxtLabel(gca,plotLabels(8));
set([gca,xl,yl,tl,txl],'fontSize',18,'fontWeight','bold');


%===================================================================== %
% Fourier series fitting                                                %
% ===================================================================== %
    
    % Number of Fourier series coefficients (an, bn) 
    % !!There are 2*n total coefficients!!
    % n for an
    % n for bn
    n = 16;
    
    % this is an outer-product
    phi = 2*pi/xMax .* (1:n)' * x;

    % basis function matrix
    fourierMatrix = [cos(phi);sin(phi)];
    
    % construct the function to fit from sinusoids and strictly-decaying
    % exponentials
    f = ( cos(2*pi* 6 * x/xMax) .* exp(-5 * abs(x)/xMax) + sin(2*pi* 3 * x/xMax) .* exp(-2 * abs(x)/xMax) );
    %f = cos(8*pi*x/xMax) .* exp(-abs(x)) + sin(5*pi*x/xMax) .* exp(-2*abs(x));
    
    % assign inputs and targets for ANN
    inputs = fourierMatrix;
    targets = f;
    
    % running the ANN
    [weights,bias,pltHandle] = fitFunctionANN_1D_studentVersion(targets,inputs,learnRateW,learnRateB,trainingIterations);

    % delete the error plotting figure
    close(ancestor(pltHandle,'figure'));
    
    % reconstruct function
    f2 = weights*inputs+bias;
    
    % calculate R^2 value
    Rsqrd = 1-sum( (targets-f2).^2 )/sum( (targets-mean(targets)).^2 );
    
    % use the sine/cosine coefficients to calculate the complex
    % coefficients: https://en.wikipedia.org/wiki/Fourier_series#Exponential_form_coefficients
    an_bn = weights;
    a0 = bias;
    
    % n>0
    cn = 1/2 * (an_bn(1:n) + 1i * an_bn(n+1:end));

    % [n<0, n=0, n>0]
    cn = [flip(conj(cn)),a0,cn];
    
    % Now, calculate same coefficients using DFT matrix instead
    DFTMat = exp(2*pi*1i*(-n:n)' * x/xMax);
    F = DFTMat * f'/N;
    
% plot ---------------------------------------------------------------- %
    
    nexttile(9)
    line(x,f,'color','r','lineWidth',2);
    line(x,f2,'color','b','marker','d');
    box on; axis tight;
    xl = xlabel('x');
    yl = ylabel('f(x)','Rotation',0);
    tl = title('Complex Exponentials');
    txlRsqrd = makeTxtLabel(gca,['  R^2: ', num2str(Rsqrd)],1);    
    txl = makeTxtLabel(gca,plotLabels(9));
    set([gca,xl,yl,tl,txl,txlRsqrd],'fontSize',18,'fontWeight','bold');
    

    nexttile(10)
    line(fx,abs(F),'color','k','lineWidth',2);
    line(fx,abs(cn),'color','b','marker','*');
    box on; axis tight;
    xl = xlabel('f_x');
    yl = ylabel('c_n','Rotation',0);
    tl = title('Complex Coefficients');
    txl = makeTxtLabel(gca,plotLabels(10));
    set([gca,xl,yl,tl,txl],'fontSize',18,'fontWeight','bold');


fprintf(['I would use either exact determined polynomials and \n' ...
    ' complex exponentials because they give a ' ...
    'R^2 of 1 \n and they have the least variation in actual and fit function coefficients'])

%Claude AI was used to help with formatting checks, and code layout/and
%printing to terminal
    
function textLabel = makeTxtLabel(ax,labelText,opt)
    
    if nargin==2
        opt=0;
    end
    
    xLim = ax().XLim;
    yLim = ax().YLim;
    xRange = max(xLim)-min(xLim);
    yRange = max(yLim)-min(yLim);
    if opt
        x0 = min(xLim) + 0.0 * xRange;
        y0 = mean(yLim) - 0.01 * yRange;
    
    else

        x0 = max(xLim) + 0.01 * xRange;
        y0 = max(yLim) - 0.01 * yRange;
    
    end
    
    textLabel = text(ax,x0,y0, labelText, 'HorizontalAlignment','left', 'VerticalAlignment','top');
    
end