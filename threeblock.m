function [mu,S,beta,timelist,objlist,iteration] = threeblock(n,p,X)
    %% initialization
    S = 2*eye(p); mu = 0.2*ones(p,1); beta = 0.5;
    S_t = 2*S; mu_t = 2*ones(p,1); beta_t = 2;

    %% check obj value
    sm = 0;
    for i = 1:n
       sm = sm + ((X(:,i) - mu)' * (S \ (X(:,i) - mu)))^beta;
    end
    obj0 = -n*(log(beta) - 0.5*p*log(2)*(beta^-1) - log(gamma(p/(2*beta))) )...
        + 0.5*n*log(det(S)) + 0.5*sm;   
 
    %% initialize data list
    t0 = cputime;
    timelist = [0.001]; objlist = [obj0]; iteration = 1;

    %% main loop
    while norm(S_t-S,'fro') > 1e-5 || abs(beta_t-beta) > 1e-5 || norm(mu_t-mu,'fro') > 1e-5
%     while norm(S_t-S,'fro') > 1e-5 || abs(beta_t-beta) > 3e-4 || norm(mu_t-mu,'fro') > 1e-5
    %% initialize
        iteration = iteration + 1;
        S_t = S; beta_t = beta; mu_t = mu;

                
    %% update mu
        % compute w
        w = zeros(n,1);
        for i = 1:n
            w(i) = ( (X(:,i) - mu_t)' * (S_t \ (X(:,i) - mu_t)) )^(beta_t-1);
        end
        
        % compute mu_t+1
        fenzi = 0; fenmu = 0;
        for i = 1:n
           fenzi = fenzi + w(i) * X(:,i);
           fenmu = fenmu + w(i);
        end
        mu =  fenzi / fenmu;
     
    %% update Sigma
        % compute w
        w = zeros(n,1);
        for i = 1:n
            w(i) = ( (X(:,i) - mu)' * (S_t \ (X(:,i) - mu)) )^(beta_t-1);
        end
        
        % compute sigma_t+1
        sm = 0;
        for i = 1:n
           sm = sm + w(i) * (X(:,i) - mu) * (X(:,i) - mu)';
        end
        S = (beta_t * sm)/n;
%         S = p*S/trace(S);
      
    %% update beta
        % compute y
        y = zeros(n,1);
        for i = 1:n
            y(i) = (X(:,i) - mu)' * (S \ (X(:,i) - mu)); 
        end 
        
        f = @(bt,y) n*( 1/bt + ( p*log(2)*((bt)^(-2)) )/2 ...
        + ( p*((bt)^(-2))*psi(p/(2*bt)) )/2 )...
        - 0.5*sum((y.^bt).*log(y));
    
        TOL = 1e-7;
        a = 0; b = 1;
        while 1
           c = (a+b)/2;
           obj_beta = f(c,y);
           if abs(obj_beta) <TOL
               beta = c;
               break
           end
           if obj_beta > 0
               a = c;
           else
               b = c;
           end
        end

      %% record cputime
        t = cputime;
        timelist = [timelist t-t0];

    %% check obj value
        sm = 0;
        for i = 1:n
           sm = sm + ((X(:,i) - mu)' * (S \ (X(:,i) - mu)))^beta;
        end
        obj = -n*(log(beta) - 0.5*p*log(2)*(beta^-1) - log(gamma(p/(2*beta))) )...
            + 0.5*n*log(det(S)) + 0.5*sm;
        objlist = [objlist obj];
    end
end