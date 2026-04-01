import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 20, 5)
.faces(">Z")
.workplane()
.moveTo(-6, 0)
.hLineTo(6, forConstruction=False)
.vLine(-2)
.hLineTo(-6)
.close()
.cutBlind(-3)
)
cq.exporters.export(result, 'GT.stl')