% PARAMETERS
v = 500;
dt = 0.01;
total_t = 10000;
steps = total_t / dt;
gamma_dot = 1;

save_interval = 500;  
frame_range = 100;    


r = zeros(4, 3);
for i = 1:4
    r(i, :) = [(i - 1) * sqrt(v), 0, 0];
end

% VIDEO
vwriter = VideoWriter('4bead_simulation.avi', 'Motion JPEG AVI');
vwriter.FrameRate = 60;
open(vwriter);
fig = figure('Visible', 'off');

% SIMULATION
for step = 1:steps
    F = zeros(4, 3);

    % Spring forces
    for i = 1:3
        Rij = r(i+1,:) - r(i,:);
        Rmag = norm(Rij);
        rhat = Rmag / v;
        if abs(1 - rhat^2) < 1e-6
            rhat = 0.999;
        end
        Fspring = ((3 - rhat^2) / (v * (1 - rhat^2))) * Rij;
        F(i,:) = F(i,:) + Fspring;
        F(i+1,:) = F(i+1,:) - Fspring;
    end

    % Brownian + shear flow
    for i = 1:4
        B = sqrt(6 / dt) * (2 * rand(1, 3) - 1);
        shear = [gamma_dot * r(i,2), 0, 0];
        r(i,:) = r(i,:) + dt * (F(i,:) + B + shear);
    end

    % FRAMES
    if mod(step, save_interval) == 0
        clf;
        plot3(r(:,1), r(:,2), r(:,3), 'o-', 'LineWidth', 2, 'MarkerSize', 6);
        COM = mean(r);
        axis([COM(1)-frame_range, COM(1)+frame_range, ...
              COM(2)-frame_range, COM(2)+frame_range, ...
              COM(3)-frame_range, COM(3)+frame_range]);
        xlabel('X'); ylabel('Y'); zlabel('Z');
        title(sprintf('t = %.1f', step*dt));
        view(3); grid on;
        drawnow;

        frame = getframe(fig);
        writeVideo(vwriter, frame);
    end
end

close(vwriter);
disp(' Video saved: 4bead_simulation.avi');
