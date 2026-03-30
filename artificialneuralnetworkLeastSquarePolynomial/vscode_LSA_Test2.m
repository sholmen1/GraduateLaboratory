function [Func, FuncLS, c, a, b, Rsqrd] = vscode_LSA(M, n)
xMax=2;
dx=xMax/M;
x=-xMax/2:dx:xMax/2-dx;

fMax= 1/dx;
dfx=fMax/M;
fx=-fMax/2:dfx:fMax/2-dfx;
[~,~]=meshgrid(x,fx);

%Generate random coefficents 
coeff= (rand(n+1,1)-0.5); %column vector

%Vandermonde matrix
X = repmat(x(:), 1, n+1) .^repmat(0:n, M, 1); 

%polynomial function
Func = X * coeff; 

%least squares fit
% Least squares fit - handle different cases
    if M > n+1
        % Overdetermined case - use normal equation
        LSquarescoeffs = (X'*X)\(X'*Func);
    elseif M == n+1
        % Exactly determined case - use direct matrix solve
        LSquarescoeffs = X\Func;
    else
        % Underdetermined case - use pseudoinverse
        LSquarescoeffs = pinv(X)*Func;
    end
  
FuncLS = X * LSquarescoeffs; %Fitted polynomial function

%calculate R^2 value
Rsqrd = 1-sum( (Func-FuncLS).^2 )/sum( (Func-mean(Func)).^2 );
a=coeff;
b=LSquarescoeffs;
c=abs(a-b);
end