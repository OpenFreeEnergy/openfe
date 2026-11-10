.. _userguide_result_networks:

Working with networks of results
================================

After running a **network** of free energy calculations,
we often want to analyse the corresponding network of results.

.. _userguide_MLE:

Converting relative results to absolute estimates
-------------------------------------------------

When a network of relative free energies has been calculated,
a commonly performed task is to transform these pairwise estimations of **relative** free energy differences (:math:`\Delta \Delta G`)
into **absolute** free energy differences (:math:`\Delta G`).
This is done using a maximum likelihood estimator (MLE) [1]_,
as implemented in the `cinnabar`_ package.
This approach uses the matrix of relative pairwise measurements and their uncertainties,
to estimate the overall ranking of ligands.
To use this approach the network of pairwise measurements needs to be fully connected,
i.e. there should be a way to trace a path along pairwise measurements between any two nodes (ligands) on the network.

.. note::
   The results of a MLE estimation will have a **mean** of 0.0,
   meaning that there will be some estimates with positive values and some estimates with negative values.
   These predictions (:math:`\Delta G_{pred}`) can be shifted to match the magnitude of the experimental data,
   to satisfy the below equation where the sum is performed over N molecules that have experimental data (:math:`\Delta G_{exp}`) [2]_.

   .. math::

      \sum_i^N \Delta G^i_{exp} = \sum_i^N \Delta G^i_{pred}

Gathering and analysing network results
----------------------------------------

The :ref:`openfe gather <cli_gather>` command collects the results of many individual calculations into a table,
either of relative free energies or of their :ref:`corresponding MLE values <userguide_MLE>`.
The `cinnabar`_ package can be used from within Python to manipulate networks of free energy estimates,
for example to plot them against experimental data.

.. admonition:: How-to guides

   * :doc:`/how_to_guides/gather_results`
   * :doc:`/how_to_guides/plotting_with_cinnabar`

See also
--------

For handling the results of a single calculation, please consult :ref:`userguide_individual_results`

.. [1] Optimal Measurement Network of Pairwise Differences, Huafeng Xu, J. Chem. Inf. Model. 2019, 59, 11, 4720-4728
.. [2] Accurate and Reliable Prediction of Relative Ligand Binding Potency in Prospective Drug Discovery by Way of a Modern Free-Energy Calculation Protocol and Force Field
       Lingle Wang, Yujie Wu, Yuqing Deng, Byungchan Kim, Levi Pierce, Goran Krilov, Dmitry Lupyan, Shaughnessy Robinson, Markus K. Dahlgren, Jeremy Greenwood, Donna L. Romero, Craig Masse, Jennifer L. Knight, Thomas Steinbrecher, Thijs Beuming, Wolfgang Damm, Ed Harder, Woody Sherman, Mark Brewer, Ron Wester, Mark Murcko, Leah Frye, Ramy Farid, Teng Lin, David L. Mobley, William L. Jorgensen, Bruce J. Berne, Richard A. Friesner, and Robert Abel
       Journal of the American Chemical Society 2015 137 (7), 2695-2703 DOI: 10.1021/ja512751q
.. _cinnabar: https://github.com/OpenFreeEnergy/cinnabar
