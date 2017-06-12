/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = 100;
  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    particle.id = i;

    random_device rd;
    default_random_engine gen(rd());
    normal_distribution<> dist_x(x, std[0]);
    normal_distribution<> dist_y(y, std[1]);
    normal_distribution<> dist_theta(theta, std[2]);

    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1;

    weights.push_back(particle.weight);
    particles.push_back(particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  for (int i = 0; i < particles.size(); i++) {
    double prediction_theta = particles.at(i).theta + yaw_rate * delta_t;

    if (fabs(yaw_rate) < 0.0001) {
      particles.at(i).x += velocity * delta_t * cos(particles.at(i).theta);
      particles.at(i).y += velocity * delta_t * sin(particles.at(i).theta);
    } else {
      particles.at(i).x += (velocity / yaw_rate) * (sin(prediction_theta) - sin(particles.at(i).theta));
      particles.at(i).y += (velocity / yaw_rate) * (cos(particles.at(i).theta) - cos(prediction_theta));;
      particles.at(i).theta = prediction_theta;
    }

    random_device rd;
    default_random_engine gen(rd());
    normal_distribution<> dist_x(particles.at(i).x, std_pos[0]);
    normal_distribution<> dist_y(particles.at(i).y, std_pos[1]);
    normal_distribution<> dist_theta(particles.at(i).theta, std_pos[2]);

    particles.at(i).x = dist_x(gen);
    particles.at(i).y = dist_y(gen);
    particles.at(i).theta = dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  for (int i = 0; i < observations.size(); i++) {
    double min_dist = -1;
    for (int j = 0; j < predicted.size(); j++) {
      double current_dist = dist(predicted.at(j).x, predicted.at(j).y, observations.at(i).x, observations.at(i).y);
      if (min_dist == -1 || current_dist < min_dist) {
        min_dist = current_dist;
        observations.at(i).id = predicted.at(j).id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html

  for (int i = 0; i < particles.size(); i++) {
    std::vector<LandmarkObs> predicted_landmarks;
    std::vector<LandmarkObs> transformed_observations;

    for (auto observation: observations) {
      LandmarkObs transformed_observation;
      transformed_observation.x = observation.x * cos(particles.at(i).theta) - observation.y * sin(particles.at(i).theta) + particles.at(i).x;
      transformed_observation.y = observation.x * sin(particles.at(i).theta) + observation.y * cos(particles.at(i).theta) + particles.at(i).y;
      transformed_observations.push_back(transformed_observation);
    }

    for (auto landmark: map_landmarks.landmark_list) {
      if (dist(landmark.x_f, landmark.y_f, particles.at(i).x, particles.at(i).y) < sensor_range) {
        LandmarkObs landmarkObs;
        landmarkObs.id = landmark.id_i;
        landmarkObs.x = landmark.x_f;
        landmarkObs.y = landmark.y_f;
        predicted_landmarks.push_back(landmarkObs);
      }
    }

    dataAssociation(predicted_landmarks, transformed_observations);
    particles.at(i).weight = 1;

    for (auto transformed_obs: transformed_observations) {
      for (auto landmark: predicted_landmarks) {
        if (landmark.id == transformed_obs.id) {
          double delta_x = landmark.x - transformed_obs.x;
          double delta_y = landmark.y - transformed_obs.y;
          double exp_value = exp(
                  -(pow(delta_x, 2) / (2 * pow(std_landmark[0], 2)) + pow(delta_y, 2) / (2 * pow(std_landmark[1], 2))));
          particles.at(i).weight *= 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]) * exp_value;
          break;
        }
      }
    }

    weights.at(i) = particles.at(i).weight;
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  random_device rd;
  default_random_engine gen(rd());
  discrete_distribution<> dist(weights.begin(), weights.end());

  vector<Particle> resampled_particles;
  for (size_t i = 0; i < num_particles; i++) {
    resampled_particles.push_back(particles.at(dist(gen)));
  }
  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
