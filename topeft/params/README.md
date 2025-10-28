### Documentation

:warning: Is this documentation out of date?
* Twiki with the b tag working points (does not have UL16 yet): https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation

### CR/SR plot configuration

* Background groups used for the control- and signal-region plots are driven by
  pattern lists defined in `params/cr_sr_plots_metadata.yml`.
* Any Monte Carlo process that fails to match one of the configured patterns is
  no longer dropped from the stack. Instead, the plotting code will emit a
  warning and create a one-off group named after the raw process. The fallback
  group participates in the stack and inherits a default colour palette so the
  total MC yield remains unchanged.
