HM := headmaster

# ── Model management ────────────────────────────────────────────

.PHONY: model-add
model-add:
	$(HM) model-add --name $(NAME) --path $(PATH) --dim $(DIM)

.PHONY: model-list
model-list:
	$(HM) model-list

.PHONY: model-activate
model-activate:
	$(HM) model-activate --name $(NAME)

.PHONY: model-remove
model-remove:
	$(HM) model-remove --name $(NAME)

# ── Embedding & training ────────────────────────────────────────

.PHONY: embed
embed:
	$(HM) embed $(if $(HEAD),--head $(HEAD))

.PHONY: train
train: embed
	$(HM) train $(if $(HEAD),--head $(HEAD))

.PHONY: status
status:
	$(HM) status $(if $(HEAD),--head $(HEAD))

# ── Classification ──────────────────────────────────────────────

.PHONY: classify
classify:
	$(HM) classify --head $(HEAD) --src $(SRC) $(if $(DEST),--dest $(DEST))

# ── Export & cleanup ────────────────────────────────────────────

.PHONY: export
export:
	$(HM) export --dest $(DEST)

.PHONY: clean
clean:
	$(HM) clean $(if $(HEAD),--head $(HEAD))
