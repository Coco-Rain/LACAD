import cadquery as cq

result = (
cq.Workplane("XY")
.box(20, 20, 2)
.faces(">Z")
.workplane()
.center(-3, -3)
.rect(4, 4, forConstruction=False)
.extrude(2)
)
cq.exporters.export(result, 'GT.stl')