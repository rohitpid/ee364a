% Problem 9-30a / A8-2

function main
clc; clear;
n = 100;
m = 200;
randn('state',1);
A=randn(m,n);
x = zeros(n,1);
grad = 100;
alpha = 0.3;
beta = 0.5;
f = [];
count = 1;
f(1) = func(x,A,m,n);
% CVX says Optimal value (cvx_optval): -144.698
    while norm(grad) >= 0.1
        t=1;
        count = count+1;
        grad = gradF(x,A,m,n); % calculate gradient
        del_x = -grad/norm(grad);
        %backtracking
        while true
            step = t*del_x;
%             x = x/max(norm(A*x),norm(x));
            threshold = f(count-1) + alpha*t*grad'*del_x;
            f(count) = func(x+step,A,m,n);
            if ~isreal(f(count))
                t = beta*t;
            else
%                 disp(['threshold: ' num2str(threshold)])
                disp(['f(count): ' num2str(f(count))])
                disp(['norm(grad): ' num2str(norm(grad))])
                if(f(count) <= threshold)
                    x=x+step;
                    break;
                end
                t = beta*t;
            end

        end
    end
    
    cvx_begin
    variable x(n);
    minimize -sum(log(1-A*x))-sum(log(1-x.^2));
    subject to
    abs(x)<=1;
    A*x<=1;
    cvx_end
end

function functionVal = func(x,A,m,n)
    functionVal = -sum(log(ones(m,1) - A*x)) - sum(log(ones(n,1) - x.^2));
end

function gradVal = gradF(x,A,m,n)
    gradVal =  A'*(ones(m,1)./(ones(m,1) - A*x)) + 2*x ./ (ones(n,1) - x .^2);
end