import cadquery as cq

result = (
cq.Workplane("XY")
.box(2, 2, 2)
.faces(">Z")
.shell(-0.2)
.faces(">Z")
.edges("not(<X or >X or <Y or >Y)")
.chamfer(0.125)
)
cq.exporters.export(result, 'GT.stl')